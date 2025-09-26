#include "Binding/LocalisationBinding.hpp"
#include "Binding/Robocup.hpp"

#include "CameraState/CameraState.hpp"

#include "Localisation/Ball/BallStackFilter.hpp"
#include "Localisation/Field/FeatureObservation.hpp"
#include "Localisation/Field/FieldObservation.hpp"
#include "Localisation/Field/RobotController.hpp"
#include "Localisation/Field/TagsObservation.hpp"
#include "Localisation/Field/GyroYawObservation.hpp"
#include <hl_monitoring/top_view_drawer.h>

#include "Utils/Drawing.hpp"
#include "Utils/Interface.h"
#include "Utils/OpencvUtils.h"

#include "scheduler/MoveScheduler.h"
#include "services/DecisionService.h"
#include "services/LocalisationService.h"
#include "services/ModelService.h"
#include "services/RefereeService.h"

#include "unistd.h"

#include <hl_communication/perception.pb.h>

#include <rhoban_utils/logging/logger.h>
#include <rhoban_utils/util.h>
#include <utility>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <cstdlib>

using namespace hl_monitoring;
using namespace rhoban_utils;
using namespace Vision::Localisation;
using namespace hl_communication;

using Vision::Utils::CameraState;

static rhoban_utils::Logger fieldLogger("RobocupFieldPF");

namespace Vision {
LocalisationBinding::LocalisationBinding(MoveScheduler* scheduler_, Robocup* vision_binding_)
    : vision_binding(vision_binding_),
      scheduler(scheduler_)
      //, nb_particles_ff(5000)
      ,
      nb_particles_ff(1000),
      robotQ(-1),
      isGoalKeeper(false),
      consistencyEnabled(true),
      consistencyScore(0)  //[Sol] was 1
      ,
      consistencyStepCost(0.005),
      consistencyBadObsCost(0.02)  // 0.02
      ,
      consistencyGoodObsGain(0.04)  // 0.1
      ,
      consistencyResetInterval(10)  // 30
      ,
      consistencyMaxNoise(5.0),
      cs(new CameraState(scheduler_))
      //, period(1.0)
      ,
      period(0.5)  // Good for birdview better fitting largenumber of observations
      //, period(0.02) //[Sol] for debug particle filter on logs each frame
      ,
      maxNoiseBoost(10.0),
      noiseBoostDuration(5),
      isForbidden(false),
      bind(nullptr),
      _runThread(nullptr),
      odometryMode(false) {
  scheduler->getServices()->localisation->setLocBinding(this);
  field_filter = new Localisation::FieldPF();

  heatMap = cv::Mat(740, 1040, CV_32FC1);  //[Sol] float image
  heatMapDirectionsMask = cv::Mat(740, 1040, CV_8UC1);
  extendedTopView = cv::Mat(740 + 100, 1040, CV_8UC3);  // 100 pixel header for status text, etc

  init();

  currTS = getNowTS();
  lastTS = currTS;
  lastFieldReset = currTS;
  lastUniformReset = currTS;

  _runThread = new std::thread(std::bind(&LocalisationBinding::run, this));
}

LocalisationBinding::~LocalisationBinding() {}

void LocalisationBinding::run() {
  while (true) {
    double startTimer = (double)cv::getTickCount();
    step();
    double elapsed_real_ms = 1000.0 * (((double)cv::getTickCount() - startTimer) / cv::getTickFrequency());
    // fieldLogger.log("Total localisation step computation time: %lf ms", elapsed_real_ms);
    if (elapsed_real_ms / 1000.0 > period) {
      fieldLogger.error(
          "Localisation step computation time greater than localisation update period! (compute time %f ms, period %f "
          "ms)",
          elapsed_real_ms, period * 1000);
    }

    // Now sleep to call step() acoording to "period" time setting (usually each 500..1000 milliseconds)
    if (scheduler->isFakeMode()) {
      while (true) {
        // Here, we check if there is a premature exit or if enough time has
        // been elapsed according to vision TimeStamps
        double elapsed = diffSec(currTS, getNowTS());
        bool referee_allow_playing = refereeAllowsToPlay();
        bool premature_exit = field_filter->isResetPending() && referee_allow_playing;
        if (elapsed > period || premature_exit) break;
        usleep(50 * 1000);
        bool isStreamingTopView = RhIO::Root.frameIsStreaming("/localisation/TopView");
        if (isStreamingTopView) {
          redrawTopViewHeader();
          publishExtendedTopViewToRhIO();  //[Sol] do it at 20fps
        }
      }
    } else {
      double elapsed = diffSec(currTS, getNowTS());
      // fieldLogger.log("Step time: %lf", elapsed);
      if (elapsed < period) {
        int sleep_us = (int)((period - elapsed) * 1000 * 1000);
        // Sleep a total time of sleep_us by small intervals and interrupt if
        // there is a reset pending
        int count = sleep_us / 50000;
        for (int k = 0; k < count; k++) {
          bool referee_allow_playing = refereeAllowsToPlay();
          bool premature_exit = field_filter->isResetPending() && referee_allow_playing;
          if (premature_exit) {
            fieldLogger.log("Premature exit from sleep (reset pending)");
            break;
          }
          usleep(50 * 1000);
          bool isStreamingTopView = RhIO::Root.frameIsStreaming("/localisation/TopView");
          if (isStreamingTopView) {
            redrawTopViewHeader();
            publishExtendedTopViewToRhIO();  //[Sol] do it at 20fps
          }
        }
      }
    }
  }
}

void LocalisationBinding::init() {
  initRhIO();
  importFromRhIO();
  field_filter->initializeAtUniformRandom(nb_particles_ff);
}

// TODO: eventually build Image handlers

void LocalisationBinding::initRhIO() {
  // Only bind once
  if (bind != nullptr) {
    return;
  }

  bind = new RhIO::Bind("localisation");

  // Init interface with RhIO
  RhIO::Root.newCommand(
      "localisation/resetFilters", "Reset all particle filters to an uniform distribution",
      [this](const std::vector<std::string>& args) -> std::string {
        lastFieldReset = getNowTS();
        currTS = lastFieldReset;
        lastUniformReset = lastFieldReset;
        vision_binding->ballStackFilter->clear();
        vision_binding->clearRememberObservations = true;
        consistencyScore = 0;
        // field_filter->askForReset();
        if (gyroYawToOpponentGoalSetted) {
          double gyroYaw;  // in degrees
          // Code below is copy-paste from float Helpers::getGyroYaw() to correctly work in real mode and fake mode
          if (scheduler->isFakeMode()) {
            gyroYaw = rhoban::frameYaw(scheduler->getServices()->model->model.selfToWorld().rotation()) / M_PI * 180.0;
          } else {
            gyroYaw = scheduler->getManager()->dev<RhAL::GY85>("imu").getGyroYaw() / M_PI * 180.0;
          }

          // field_filter->gyroYawForUniformDirectedReset = gyroYawToOpponentGoal;
          rhoban_utils::Angle gyroYawToOpponentGoalAngle = rhoban_utils::Angle(gyroYawToOpponentGoal);
          rhoban_utils::Angle gyroYawAngle = rhoban_utils::Angle(gyroYaw);
          rhoban_utils::Angle yawAngle =
              gyroYawAngle - gyroYawToOpponentGoalAngle;  // Should give 0 when robot is facing opponent's goal
          double yaw = yawAngle.getSignedValue();
          field_filter->gyroYawForUniformDirectedReset = yaw;

          field_filter->askForReset(FieldPF::ResetType::UniformDirected);
        } else {
          //field_filter->askForReset(FieldPF::ResetType::Uniform);
          const auto& repr_particle = field_filter->getRepresentativeParticle();
          field_filter->gyroYawForUniformDirectedReset = repr_particle.getOrientation().getSignedValue();
          field_filter->askForReset(FieldPF::ResetType::UniformDirected);          
        }
        return "Field have been reset";
      });
  RhIO::Root.newCommand("localisation/bordersReset", "Reset on the borders",
                        [this](const std::vector<std::string>& args) -> std::string {
                          fieldReset(FieldPF::ResetType::Borders);
                          return "Field have been reset";
                        });
  RhIO::Root.newCommand("localisation/fallReset", "Apply a fall event on field particle filter",
                        [this](const std::vector<std::string>& args) -> std::string {
                          fieldReset(FieldPF::ResetType::Fall);
                          return "Field have been reset";
                        });
  RhIO::Root.newCommand(
      "localisation/customReset", "Reset the field particle filter at the custom position with custom noise [m,deg]",
      [this](const std::vector<std::string>& args) -> std::string {
        unsigned int k = 0;

        auto rhioNode = &(RhIO::Root.child("/localisation/field/fieldPF"));
        for (std::string item : {"customX", "customY", "customTheta", "customNoise", "customThetaNoise"}) {
          if (args.size() > k) {
            rhioNode->setFloat(item, atof(args[k].c_str()));
          }
          k++;
        }
        lastFieldReset = getNowTS();
        currTS = lastFieldReset;
        consistencyScore = 1;
        field_filter->askForReset(FieldPF::ResetType::Custom);
        return "Field have been reset";
      });

  //[Sol]
  RhIO::Root.newCommand(
      "localisation/tareGyroToOpGoalCustom",
      "Saves argument [deg] IMU yaw angle as a angle to opponent goal for robust particle filter performance",
      [this](const std::vector<std::string>& args) -> std::string {
        if (args.size() == 1) {
          float a = atof(args[0].c_str());
          gyroYawToOpponentGoal = a;
          gyroYawToOpponentGoalSetted = true;
          return "gyroYawToOpponentGoal updated OK";
        } else {
          return "gyroYawToOpponentGoal NOT updated - wrong arguments count!";
        }
      });

  RhIO::Root.newCommand(
      "localisation/tareGyroToOpGoal",
      "Saves current IMU yaw angle as a angle to opponent goal for robust particle filter performance",
      [this](const std::vector<std::string>& args) -> std::string {
        double _gyroYaw;
        if (scheduler->isFakeMode()) {
          _gyroYaw = rhoban::frameYaw(scheduler->getServices()->model->model.selfToWorld().rotation()) / M_PI * 180.0;
        } else {
          _gyroYaw = scheduler->getManager()->dev<RhAL::GY85>("imu").getGyroYaw() / M_PI * 180.0;
        }
        gyroYawToOpponentGoal = _gyroYaw;
        gyroYawToOpponentGoalSetted = true;
        return "gyroYawToOpponentGoal updated OK";
      });

  RhIO::Root.newCommand("localisation/resetGyroToOpGoal",
                        "Clears IMU yaw angle to opponent goal and disable it's processing in particle filter",
                        [this](const std::vector<std::string>& args) -> std::string {
                          gyroYawToOpponentGoalSetted = false;
                          return "gyroYawToOpponentGoal reset OK";
                        });

  // Number of particles in the field filter
  bind->bindNew("field/nbParticles", nb_particles_ff, RhIO::Bind::PullOnly)
      ->defaultValue(nb_particles_ff)
      ->comment("Number of particles in the localisation filter");
  bind->bindNew("field/odometryMode", odometryMode, RhIO::Bind::PullOnly)
      ->defaultValue(odometryMode)
      ->comment("Is the localization based only on odometry?");
  // consistency
  bind->bindNew("consistency/enabled", consistencyEnabled, RhIO::Bind::PullOnly)
      ->defaultValue(consistencyEnabled)
      ->comment("Is consistency check enabled? (If disable, consistencyScore is not updated)");
  bind->bindNew("consistency/elapsedSinceReset", elapsedSinceReset, RhIO::Bind::PushOnly)
      ->defaultValue(0)
      ->comment("Elapsed time since last reset (from any source) [s]");
  bind->bindNew("consistency/elapsedSinceUniformReset", elapsedSinceUniformReset, RhIO::Bind::PushOnly)
      ->defaultValue(0)
      ->comment("Elapsed time since last uniform reset (from any source) [s]");
  bind->bindNew("consistency/score", consistencyScore, RhIO::Bind::PushOnly)
      ->defaultValue(consistencyScore)
      ->maximum(1.0)
      ->minimum(0.0)
      ->comment("Current consistency quality");
  bind->bindNew("consistency/stepCost", consistencyStepCost, RhIO::Bind::PullOnly)
      ->defaultValue(consistencyStepCost)
      ->comment("The reduction of consistencyScore at each step");
  bind->bindNew("consistency/badObsCost", consistencyBadObsCost, RhIO::Bind::PullOnly)
      ->defaultValue(consistencyBadObsCost)
      ->comment("The reduction of consistencyScore for each bad observation");
  bind->bindNew("consistency/goodObsGain", consistencyGoodObsGain, RhIO::Bind::PullOnly)
      ->defaultValue(consistencyGoodObsGain)
      ->comment("The increase of consistencyScore for each 'good' observation");
  bind->bindNew("consistency/resetInterval", consistencyResetInterval, RhIO::Bind::PullOnly)
      ->defaultValue(consistencyResetInterval)
      ->comment("The minimal time to wait between two consistency resets [s]");
  bind->bindNew("consistency/maxNoise", consistencyMaxNoise, RhIO::Bind::PullOnly)
      ->defaultValue(consistencyMaxNoise)
      ->comment("Noise factor at 0 consistencyScore");
  bind->bindNew("period", period, RhIO::Bind::PullOnly)
      ->defaultValue(period)
      ->maximum(30.0)
      ->minimum(0.0)
      ->comment("Period between two ticks from the particle filter");
  bind->bindNew("consistency/elapsedSinceConvergence", elapsedSinceConvergence, RhIO::Bind::PushOnly)
      ->defaultValue(0)
      ->comment("Elapsed time since last convergence or reset [s]");
  bind->bindNew("field/maxNoiseBoost", maxNoiseBoost, RhIO::Bind::PullOnly)
      ->defaultValue(maxNoiseBoost)
      ->maximum(30.0)
      ->minimum(1.0)
      ->comment("Maximal multiplier for exploration in boost mode");
  bind->bindNew("field/noiseBoostDuration", noiseBoostDuration, RhIO::Bind::PullOnly)
      ->defaultValue(noiseBoostDuration)
      ->maximum(30.0)
      ->minimum(0.0)
      ->comment("Duration of the noise boost after global reset [s]");
  bind->bindNew("debugLevel", debugLevel, RhIO::Bind::PullOnly)
      ->defaultValue(0)
      ->comment("Verbosity level for Localisation: 0 -> silent");

  RhIO::Root.newFrame("localisation/TopView", "Top view");
  RhIO::Root.newFrame("localisation/HeatMap", "Heat map");  //[Sol]

  // Binding Localisation items
  RobotController::bindWithRhIO();
  FeatureObservation::bindWithRhIO();
  TagsObservation::bindWithRhIO();
}

void LocalisationBinding::importFromRhIO() {
  RobotController::importFromRhIO();
  FeatureObservation::importFromRhIO();
  TagsObservation::importFromRhIO();
  field_filter->importFromRhIO();

  bind->pull();
}

void LocalisationBinding::publishToRhIO() {
  bind->push();

  field_filter->publishToRhIO();

  /*bool isStreaming = RhIO::Root.frameIsStreaming("/localisation/TopView");
  if (isStreaming)
  {
    fieldLogger.log("[Sol] Publishing TopView");
    int width = 1040;
    int height = 740;
    cv::Mat topView = getTopView(width, height);
    //cv::imshow("TopView", topView); //Dirty hack to test is topView not emply.
    RhIO::Root.framePush("/localisation/TopView", topView);
  }*/
  redrawTopViewBody();

  bool isStreamingHeatMap = RhIO::Root.frameIsStreaming("/localisation/HeatMap");
  if (isStreamingHeatMap) {
    int width = 1040;
    int height = 740;
    cv::Mat _heatMap = getHeatMap(width, height);
    RhIO::Root.framePush("/localisation/HeatMap", _heatMap);
  }
}

void LocalisationBinding::publishExtendedTopViewToRhIO() {
  bool isStreamingTopView = RhIO::Root.frameIsStreaming("/localisation/TopView");
  if (isStreamingTopView) {
    /*int width = 1040;
    int height = 740;
    cv::Mat topView = getTopView(width, height);
    RhIO::Root.framePush("/localisation/TopView", width, height+100, topView.data, width * height * 3);*/
    int width = 1040;
    int height = 740 + 100;
    RhIO::Root.framePush("/localisation/TopView", extendedTopView);
  }
}

void LocalisationBinding::step() {
  importFromRhIO();

  currTS = getNowTS();
  cs->updateInternalModel(currTS);
  //std::cout << "WOW loc Current TS: " << currTS.getTimeSec() << std::endl;

  elapsedSinceReset = diffSec(lastFieldReset, currTS);
  elapsedSinceUniformReset = diffSec(lastUniformReset, currTS);

  // Always steal informations from vision
  stealFromVision();

  // Get information from the referee
  bool refereeAllowTicks = refereeAllowsToPlay();

  // When the robot is penalized do not update anything, but increase reactivity
  if (!refereeAllowTicks) {
    lastForbidden = currTS;
    isForbidden = true;
    if (debugLevel > 0) {
      fieldLogger.log("Referee forbid ticks");
    }
    // Avoid having a uniform reset pending when robot is penalized or in initial phase
    field_filter->cancelPendingReset(FieldPF::ResetType::Uniform);

    FieldPF::ResetType pending_reset = field_filter->getPendingReset();
    if (pending_reset == FieldPF::ResetType::Custom) {
      field_filter->applyPendingReset();
    }

    if (pending_reset == FieldPF::ResetType::Borders) {
      //We are in penalized state and waiting for border resets.
      //Dont't accumulate odometry diff in this state, it will be erroneous
      lastTS = currTS;
    }

    if (true) {
      //If we are in penalized state (referee forbids us to play)
      //Let's clear the odometry integration interval not to cause exploration with time of penalisation time
      lastTS = currTS;
      lastFieldReset = currTS;

      //Also let's delete all observations
      ObservationVector observations = extractObservations();
      for (size_t id = 0; id < observations.size(); id++) {
        delete (observations[id]);
      }
    }


    importFiltersResults();
    publishToLoc();
    publishToRhIO();
    return;
  } 

  // Determining if the robot is fallen
  DecisionService* decisionService = scheduler->getServices()->decision;
  if (decisionService->isFallen) {
    if (debugLevel > 0) {
      fieldLogger.log("Robot is fallen, forbidding ticks");
    }
    publishToRhIO();
    return;
  }

  FieldPF::ResetType pending_reset = field_filter->getPendingReset();
  double elapsed_since_forbidden = diffSec(lastForbidden, currTS);
  double start_without_reset_delay = 10;  //[s]: to free the robot if it is not allowed to play
  // Wait a proper reset for some time
  // (avoid starting a tick before receiving informations from 'robocup' move)
  if (isForbidden && elapsed_since_forbidden < start_without_reset_delay &&
      (pending_reset == FieldPF::ResetType::None || pending_reset == FieldPF::ResetType::Uniform)) {
    std::ostringstream msg;
    msg << "Delaying restart of filter: "
        << "elapsed since forbidden:" << elapsed_since_forbidden << " "
        << "Pending reset: '" << FieldPF::getName(pending_reset) << "'";
    if (debugLevel > 0) {
      fieldLogger.log(msg.str().c_str());
    }

    importFiltersResults();
    publishToLoc();
    publishToRhIO();
    return;
  }

  isForbidden = false;

  if (debugLevel > 0) {
    fieldLogger.log("consistency: %d", consistencyEnabled);
  }

  // Compute observations if there is no reset pending
  ObservationVector observations;
  if (!field_filter->isResetPending() && !odometryMode) {
    observations = extractObservations();
  }

  // Update filter with the provided observations
  updateFilter(observations);

  // Update consistency
  if (consistencyEnabled && !odometryMode) {
    applyWatcher(observations);
  } else {
    consistencyScore = 1.0;
  }

  //[Sol] update heat map data
  updateHeatMap(observations);

  // Avoid memory leaks
  for (size_t id = 0; id < observations.size(); id++) {
    delete (observations[id]);
  }

  importFiltersResults();

  publishToLoc();
  publishToRhIO();
}

TimeStamp LocalisationBinding::getNowTS() {
  if (scheduler->isFakeMode()) {
    return vision_binding->sourceTS;
  }
  return TimeStamp::now();
}

std::vector<FeatureObservation*> LocalisationBinding::extractFeatureObservations() {
  std::vector<FeatureObservation*> featureObservations;
  for (const auto& entry : *features) {
    Field::POIType poiType = entry.first;
    for (const cv::Point3f& feature_pos_in_world : entry.second) {
      // TODO: consider possible case of 3d features
      cv::Point2f pos_in_self = cs->getPosInSelf(cv::Point2f(feature_pos_in_world.x, feature_pos_in_world.y));
      double robotHeight = cs->getHeight();

      rhoban_geometry::PanTilt panTiltToFeature = cs->panTiltFromXY(pos_in_self, robotHeight);
      FeatureObservation* newObs = new FeatureObservation(poiType, panTiltToFeature, robotHeight);
      // Adding new observation or merging based on similarity
      bool has_similar = false;
      for (FeatureObservation* featureObs : featureObservations) {
        if (FeatureObservation::isSimilar(*newObs, *featureObs)) {
          has_similar = true;
          featureObs->merge(*newObs);
        }
      }
      if (has_similar) {
        delete (newObs);
      } else {
        featureObservations.push_back(newObs);
      }
    }
  }

  return featureObservations;
}

std::vector<WhiteLinesCornerObservation*> LocalisationBinding::extractWhiteLinesCornerObservations() {
  // [Sol] Arena corner observations can be of two types - with and without corner. (Without corner is a border
  // observation, it's not very clear from this naming)
  std::vector<WhiteLinesCornerObservation*> whiteLinesCornerObservations;
  if (debugLevel > 0) {
    // fieldLogger.log("retrieving WhiteLinesCornerObs : %d", (int)whitelines_data.size());
  }
  for (size_t i = 0; i < whitelines_data.size(); i++) {
    if (whitelines_data[i].is_obs_valid()) {
      cv::Point2f pos_in_self = cs->getPosInSelf(whitelines_data[i].getCornerInWorldFrame());
      double robotHeight = cs->getHeight();

      rhoban_geometry::PanTilt panTiltToGoal = cs->panTiltFromXY(pos_in_self, robotHeight);
      Angle panToGoal = panTiltToGoal.pan;
      Angle tiltToGoal = panTiltToGoal.tilt;

      try {
        WhiteLinesCornerObservation* newObs =
            new WhiteLinesCornerObservation(whitelines_data[i], panToGoal, tiltToGoal, robotHeight);
        whiteLinesCornerObservations.push_back(newObs);
      } catch (const std::string msg) {
        fieldLogger.error("WhiteLinesCornerObservation inconsistency; error at construction : %s", msg.c_str());
      }
    } else {
      if (debugLevel > 0) {
        fieldLogger.log("WhiteLinesCornerObs : observation not valid");
      }
    }
  }
  if (debugLevel > 0) {
    // fieldLogger.log("returning data from extractWhiteLinesCornerObservations : %d",
    // (int)whiteLinesCornerObservations.size());
  }
  return whiteLinesCornerObservations;
}

std::vector<TagsObservation*> LocalisationBinding::extractTagsObservations() {
  std::vector<TagsObservation*> tagsObservations;
  std::map<int, std::vector<Eigen::Vector3d>> tagsInSelf;
  for (size_t markerId = 0; markerId < markerIndices.size(); markerId++) {
    Eigen::Vector3d pos_in_world = markerPositions[markerId];
    Eigen::Vector3d pos_in_self = cs->getSelfFromWorld(pos_in_world);
    tagsInSelf[markerIndices[markerId]].push_back(pos_in_self);
  }
  for (const std::pair<int, std::vector<Eigen::Vector3d>>& entry : tagsInSelf) {
    int nb_obs = entry.second.size();
    // Compute mean
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& pos : entry.second) {
      mean += pos;
    }
    mean /= nb_obs;
    // Compute stddev
    Eigen::Vector3d err2 = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& pos : entry.second) {
      Eigen::Vector3d diff = pos - mean;
      err2 += diff.cwiseProduct(diff);
    }
    Eigen::Vector3d dev = (err2 / nb_obs).cwiseSqrt();
    cv::Point3f cv_pos = Utils::eigenToCV(mean);
    cv::Point3f cv_dev = Utils::eigenToCV(dev);
    tagsObservations.push_back(new TagsObservation(entry.first, cv_pos, cv_dev, cs->getHeight(), entry.second.size()));
  }
  return tagsObservations;
}

void LocalisationBinding::stealFromVision() {
  // Declaring local unused variables to fit signature
  std::vector<std::pair<float, float>> markerCenters;
  std::vector<std::pair<float, float>> markerCentersUndistorded;
  // Stealing data
  double tagTimestamp = 0;  // Unused
  features = vision_binding->stealFeatures();
  whitelines_data = vision_binding->stealWhiteLines();  //[Sol]
  vision_binding->stealTags(markerIndices, markerPositions, markerCenters, markerCentersUndistorded, &tagTimestamp);
  if (debugLevel > 0) {
    std::ostringstream oss;
    int total_observations = 0;
    for (const auto& entry : *features) {
      int nb_obs = entry.second.size();
      total_observations += nb_obs;
      oss << nb_obs << " " << Field::poiType2String(entry.first) << ",";
    }
    total_observations += markerPositions.size();
    oss << markerPositions.size() << " marker";
    oss << "," << whitelines_data.size() << " WhiteLines[Sol]";
    fieldLogger.log("Nb observations stolen: %d (%s)", total_observations, oss.str().c_str());
  }
}

LocalisationBinding::ObservationVector LocalisationBinding::extractObservations() {
  // Declaration of the vectors used
  ObservationVector fieldObservations;

  //[Sol] Choose which data to use in particle filter here

  int obsId = 0;
  /*
    for (FeatureObservation* obs : extractFeatureObservations())
    {
      fieldObservations.push_back(obs);
      if (debugLevel > 0)
      {
        cv::Point3f pos;
        if (obs->getSeenDir(&pos))
        {
          fieldLogger.log("Feature %d of type %s -> pan: %lf, tilt: %lf, weight: %1lf, pos: %lf, %lf, %lf", obsId,
                          obs->getPOITypeName().c_str(), obs->panTilt.pan.getSignedValue(),
                          obs->panTilt.tilt.getSignedValue(), obs->weight, pos.x, pos.y, pos.z);
        }
        else
        {
          fieldLogger.error("Failed to find score for feature %d of type %s -> pan: %lf, tilt: %lf, weight: %1lf",
    obsId, obs->getPOITypeName().c_str(), obs->panTilt.pan.getSignedValue(), obs->panTilt.tilt.getSignedValue(),
    obs->weight);
        }
      }
      obsId++;
    }
  */
  for (TagsObservation* obs : extractTagsObservations()) {
    fieldObservations.push_back(obs);
    if (debugLevel > 0) {
      fieldLogger.log(
          "Tags %d -> id: %d, pos: (%.3lf, %.3lf, %.3lf), "
          "dev: (%.3lf, %.3lf, %.3lf), height: %lf  weight: %lf",
          obsId, obs->id, obs->seenPos.x, obs->seenPos.y, obs->seenPos.z, obs->stdDev.x, obs->stdDev.y, obs->stdDev.z,
          obs->robotHeight, obs->weight);
    }
    obsId++;
  }

  for (WhiteLinesCornerObservation* obs : extractWhiteLinesCornerObservations()) {
    fieldObservations.push_back(obs);
    if (debugLevel > 0) {
      fieldLogger.log("White Line/Corner %d -> pan: %lf, tilt: %lf, weight: %1lf, dist: %lf", obsId,
                      obs->getPan().getSignedValue(), obs->getTilt().getSignedValue(), obs->getWeight(),
                      obs->getBrutData().getRobotCornerDist());
    }
    obsId++;
  }

  // Add field observation, but only if we have some other observations
  if (fieldObservations.size() > 0) {
    fieldObservations.push_back(new FieldObservation(isGoalKeeper));
  }

  if (gyroYawToOpponentGoalSetted) {
    double gyroYaw;  // in degrees
    if (scheduler->isFakeMode()) {
      gyroYaw = rhoban::frameYaw(scheduler->getServices()->model->model.selfToWorld().rotation()) / M_PI * 180.0;
    } else {
      gyroYaw = scheduler->getManager()->dev<RhAL::GY85>("imu").getGyroYaw() / M_PI * 180.0;
    }
    rhoban_utils::Angle gyroYawToOpponentGoalAngle = rhoban_utils::Angle(gyroYawToOpponentGoal);
    rhoban_utils::Angle gyroYawAngle = rhoban_utils::Angle(gyroYaw);
    rhoban_utils::Angle yawAngle = gyroYawAngle - gyroYawToOpponentGoalAngle;
    fieldObservations.push_back(new GyroYawObservation(yawAngle));
  }

  return fieldObservations;
}

void LocalisationBinding::updateFilter(
    const std::vector<rhoban_unsorted::Observation<Localisation::FieldPosition>*>& obs) {
  double timer = (double)cv::getTickCount();

  ModelService* model_service = scheduler->getServices()->model;

  // ComputedOdometry
  double odom_start = lastTS.getTimeMS() / 1000.0;
  double odom_end = currTS.getTimeMS() / 1000.0;
  double elapsed = diffSec(lastTS, currTS);
  FieldPF::ResetType pending_reset = field_filter->getPendingReset();
  // If a reset has been asked, use odometry since reset.
  // Specific case for fall_reset, we still want to use the odometry prior to the reset
  if (pending_reset != FieldPF::ResetType::None && pending_reset != FieldPF::ResetType::Fall) {
    // Specific case for resets, we don't want to integrate motion before reset
    odom_start = lastFieldReset.getTimeMS() / 1000.0;
  }

  Eigen::Vector3d odo = model_service->odometryDiff(odom_start, odom_end);
  cv::Point2f robotMove;
  robotMove.x = odo(0);
  robotMove.y = odo(1);
  double orientationChange = rad2deg(odo(2));
  if (std::fabs(orientationChange) > 5) {
    fieldLogger.warning("unlikely orientation change received from odometry: %f deg", orientationChange);
  }

  // Use a boost of noise after an uniformReset
  double noiseGain = 1;
  if (odometryMode) {
    noiseGain = std::pow(10, -6);
  } else if (elapsedSinceUniformReset < noiseBoostDuration) {
    double ratio = elapsedSinceUniformReset / noiseBoostDuration;
    noiseGain = maxNoiseBoost * (1 - ratio) + ratio;
    if (debugLevel > 0) {
      fieldLogger.log("Using noise boost gain: %lf (%lf[s] elapsed)", noiseGain, elapsedSinceUniformReset);
    }
  } else if (consistencyEnabled) {
    noiseGain = 1 + (1 - consistencyScore) * (consistencyMaxNoise - 1);
    if (debugLevel > 0) {
      fieldLogger.log("Using consistency boost gain: %lf (score: %lf)", noiseGain, consistencyScore);
    }
  }

  RobotController rc(cv2rg(robotMove), orientationChange, noiseGain);

  double max_step_time = 5;  // Avoiding to have a huge exploration which causes errors
  if (elapsed > max_step_time) {
    fieldLogger.warning("Large time elapsed in fieldFilter: %f [s]", elapsed);
  }
  filterMutex.lock();
  field_filter->resize(nb_particles_ff);
  field_filter->step(rc, obs, std::min(max_step_time, elapsed));
  filterMutex.unlock();

  // If we updated the filter, it is important to update lastTS for next odometry.
  // If we skipped the step, it means that there is no point in using odometry from
  // lastTS to currTS, therefore, we can safely update lastTS
  lastTS = currTS;

  double elapsed_real = 1000.0 * (((double)cv::getTickCount() - timer) / cv::getTickFrequency());
  if (debugLevel > 0) {
    fieldLogger.log("Particle filter computation time: %lf ms", elapsed_real);
  }
}

void LocalisationBinding::publishToLoc() {
  LocalisationService* loc = scheduler->getServices()->localisation;

  // update the loc service
  cv::Point2d c = field_filter->getCenterPositionInSelf();
  Angle o = field_filter->getOrientation();

  loc->setPosSelf(Eigen::Vector3d(c.x, c.y, 0), deg2rad(o.getValue()), robotQ, consistencyScore, consistencyEnabled);

  loc->setClusters(field_filter->getPositionsFromClusters());
}

void LocalisationBinding::applyWatcher(
    const std::vector<rhoban_unsorted::Observation<Localisation::FieldPosition>*>& obs) {
  double timer = (double)cv::getTickCount();

  // Hardcoded choose of consistency methods. Seee description below
  bool useMultiplicativeConsistency = false;
  bool useMaxPotConsistency = true;

  // Apply HighLevel PF
  // double stepDeltaScore = -consistencyStepCost;
  stepDeltaScore = -consistencyStepCost;
  const auto& particle = field_filter->getRepresentativeParticle();
  std::vector<rhoban_unsorted::BoundedScoreObservation<FieldPosition>*> castedObservations;
  int obsId = 0;
  // std::cout << "Observation size = " << obs.size() << std::endl;
  consistencyObsSize = obs.size();
  consistencyGoodObsCount = 0;
  consistencyBadObsCount = 0;

  maxPotential = getMaxPotential(obs);  // Get the best aviable potential in the filed _with current orientation_

  finalReprParticlePotential = 1.0;
  for (rhoban_unsorted::Observation<FieldPosition>* o : obs) {
    double thisObsPot = o->potential(particle);
    // double thisObsPot = 2.0; //fake value

    // WhiteLinesCornerObservation* wlcObs = dynamic_cast<WhiteLinesCornerObservation*>(o);
    // if (wlcObs != nullptr) thisObsPot = wlcObs->potential(particle);

    // GyroYawObservation* gyObs = dynamic_cast<GyroYawObservation*>(o);
    // if (gyObs != nullptr) thisObsPot = gyObs->potential(particle);

    // FieldObservation* fObs = dynamic_cast<FieldObservation*>(o);
    // if (gyObs != nullptr) thisObsPot = fObs->potential(particle);

    // if(thisObsPot < 2.0) {
    finalReprParticlePotential *= thisObsPot;
    //}

    //[Sol] Original additive consistency calculation method from Rhoban's team 2019 release for DNN line corners
    // extraction
    /*
    FeatureObservation* featureObs = dynamic_cast<FeatureObservation*>(o);
    // Ignore non feature observations for quality check
    if (featureObs != nullptr)
    {
      //continue;
    // Checking Score of the particle
    double score = featureObs->potential(particle, true);
    double minScore = featureObs->getMinScore();

    if (score > minScore)
    {
      stepDeltaScore += consistencyGoodObsGain;
      consistencyGoodObsCount++;
    }
    else
    {
      stepDeltaScore -= consistencyBadObsCost;
      consistencyBadObsCount++;
    }

    // Debug
    if (debugLevel > 0)
    {
      fieldLogger.log("Observation %d: %s -> score: %f , minScore: %f", obsId, featureObs->toStr().c_str(), score,
                      minScore);
    }
    }
    */

    //[Sol] Original additive consistency calculation method from Rhoban's team 2018 release
    // We count number of "good" observations (wich supports current rept.particle)
    // and number of "bad" observations (which doesn't support curent repr.partcicle). False observations will be "bad"
    // here If total number of "good" is bigger than "bad" (with some coefficient) - let's increase consistency and
    // decrease otherwise. Drawback: when there is "half field jump" or similar case (repr.particle position is shifted
    // by half of the field, majority of observations still supports it (middle line will support goalzone line, etc))
    // the only observation which shows us that position is wrong (e.g. single corner of a goal zone) will be treated as
    // "bad" and because of other observations is still "good" - consistensy delta will be positive even when position
    // is half field wrong (which we faced on Asia-Pacific 2019)
    WhiteLinesCornerObservation* wlObs = dynamic_cast<WhiteLinesCornerObservation*>(o);
    if (wlObs != nullptr) {
      // Checking Score of the particle
      double score = wlObs->potential(particle, true);
      double minScore = 0.75;  // 0.75 //TODO: remove hardcode
      if (score > minScore) {
        // Using only corners (not lines) observations for consistency positive update, not lines
        // lines gives a lot of false goodObs giving good consistency on bad localisation)
        if (wlObs->getBrutData().hasCorner) {
          if ((useMultiplicativeConsistency == false) && (useMaxPotConsistency == false)) {
            stepDeltaScore += consistencyGoodObsGain;
          }
          consistencyGoodObsCount++;
        }
      } else {
        if ((useMultiplicativeConsistency == false) && (useMaxPotConsistency == false)) {
          stepDeltaScore -= consistencyBadObsCost;
        }
        consistencyBadObsCount++;
      }
      // Debug      //if (debugLevel > 0) {
      // fieldLogger.log("White linse corner/line observation %d: %s -> score: %f , minScore: %f", obsId,
      // wlObs->toStr().c_str(), score, minScore); printf("White linse corner/line observation %d: %s ->
      // repr.potential=%f, min.potential=%f, stepDeltaScore=%f\r\n", obsId, wlObs->toStr().c_str(), score, minScore,
      // stepDeltaScore);
      //}
    }

    // Count total number of observations for all types of observations
    obsId++;
  }

  //[Sol] Multiplicative consistency calculation method
  // Idea is: if repr.particle potential is good - we assume that observations support it, so increase consistency
  // if repr.rarticle potential is bad - observations telling us that repr.particle is in wrong position, so decrease
  // potential For example single bad conrner observation will decrease repr.potential to 0.5, two bad corners to 0.25,
  // etc So consistency here will be decreased on "half field jump" event - this is the main difference from "additive"
  // consistency Drawback: when there are many false observations - they will make repr.potential low even when
  // repr.particle is in the right position
  if (useMultiplicativeConsistency == true) {
    double minReprParticlePotential = 0.2;
    if (obsId > 0) {  // is there are some observations
      stepDeltaScore = (finalReprParticlePotential - minReprParticlePotential) * period;
    } else {
      // No observations
      stepDeltaScore = -consistencyStepCost;
    }
  }

  //[Sol] Multiplicative consistency with maxPotential calculation method
  // Idea is: when observations show that there is some cluster on filed with good potential, we should:
  // 1. increase consistency when repr.particle is in this cluster,
  // 2. decrease consistency when repr.particle in not in this cluster
  // 3. don't touch consistency when there is no good clusters
  // p.3 is this is the main dofference from methods above - when there are many false detections they will destroy any
  // clusters of good potential, so maxPotential will be almost zero and consistency will not be altered - we will rely
  // on odometry for some time
  if (useMaxPotConsistency == true) {
    if (obsId > 0) {  // is there are some observations
      if (finalReprParticlePotential > maxPotential * 0.9) {
        // Current repr.potentail is good (good observations) or maxPotential is low (many false observations)
        double gyp = getGyroYawPotential();
        if ((gyp >= 0) && (gyp < 0.45)) {
          // maxPotential is low because particle filter diverged to wrong direction (i.e. 90/180 degs shift)
          stepDeltaScore = -0.15;
        } else {
          // a) Current repr.potentail is good (good observations) or b) maxPotential is low (many false observations,
          // but gyroObs is OK) So this will raise consistency on a), and left intact on b)
          stepDeltaScore = maxPotential * 0.25;
        }
      } else {
        // Current repr.potential is bad, and there is some other point on field in which potential is good (equals
        // maxPot). This means that particle filter diverged to wrong position
        stepDeltaScore = (finalReprParticlePotential - maxPotential) * 0.35;
      }
      // In both cases positive or negative delta will be almost zero when maxPotential is low (many false observations)
      // - so consistency will not be altered
    } else {
      // No observations
      stepDeltaScore = -consistencyStepCost;
    }
  }

  double elapsed = 1000.0 * (((double)cv::getTickCount() - timer) / cv::getTickFrequency());

  consistencyScore += stepDeltaScore;
  consistencyScore = std::min(1.0, std::max(0.0, consistencyScore));
  if (debugLevel > 0) {
    fieldLogger.log("Updating consistency: computation time: %f ms | deltaStep: %f | new consistency: %f", elapsed,
                    stepDeltaScore, consistencyScore);
  }

  /// Reset of the particle filter requires several conditions
  /// - We have not reseted the filter for  long time
  /// - ConsistencyScore has reached 0
  /// - There is no reset pending on the robot
  bool resetAllowed = elapsedSinceUniformReset > consistencyResetInterval;
  bool lowConsistency = consistencyScore <= 0;
  if (debugLevel > 0) {
    fieldLogger.log("resetAllowed: %d, consistency: %f (elapsed since UR: %f)", resetAllowed, consistencyScore,
                    elapsedSinceUniformReset);
  }
  if (resetAllowed && lowConsistency && !field_filter->isResetPending()) {
    lastFieldReset = getNowTS();
    lastUniformReset = lastFieldReset;
    // consistencyScore starts at 0
    consistencyScore = 0;
    // field_filter->askForReset(FieldPF::ResetType::Uniform); //Works bad in case of 1000 particles
    if (gyroYawToOpponentGoalSetted) {
      double gyroYaw;  // in degrees
      // Code below is copy-paste from float Helpers::getGyroYaw() to correctly work in real mode and fake mode
      if (scheduler->isFakeMode()) {
        gyroYaw = rhoban::frameYaw(scheduler->getServices()->model->model.selfToWorld().rotation()) / M_PI * 180.0;
      } else {
        gyroYaw = scheduler->getManager()->dev<RhAL::GY85>("imu").getGyroYaw() / M_PI * 180.0;
      }
      // field_filter->gyroYawForUniformDirectedReset = gyroYawToOpponentGoal;
      rhoban_utils::Angle gyroYawToOpponentGoalAngle = rhoban_utils::Angle(gyroYawToOpponentGoal);
      rhoban_utils::Angle gyroYawAngle = rhoban_utils::Angle(gyroYaw);
      rhoban_utils::Angle yawAngle =
          gyroYawAngle - gyroYawToOpponentGoalAngle;  // Should give 0 when robot is facing opponent's goal
      double yaw = yawAngle.getSignedValue();
      field_filter->gyroYawForUniformDirectedReset = yaw;

      field_filter->askForReset(FieldPF::ResetType::UniformDirected);
    } else {
      //field_filter->askForReset(FieldPF::ResetType::Uniform);
      const auto& repr_particle = field_filter->getRepresentativeParticle();
      field_filter->gyroYawForUniformDirectedReset = repr_particle.getOrientation().getSignedValue();
      field_filter->askForReset(FieldPF::ResetType::UniformDirected);      
    }

    if (debugLevel > 0) {
      std::ostringstream msg;
      msg << "Asking for a full reset: " << std::endl;
      msg << "consistencyScore: " << consistencyScore << " robotQ: " << robotQ;
      fieldLogger.log(msg.str().c_str());
    }
  }
}

void LocalisationBinding::updateHeatMap(
    const std::vector<rhoban_unsorted::Observation<Localisation::FieldPosition>*>& obs) {
  bool isStreamingHeatMap = RhIO::Root.frameIsStreaming("/localisation/HeatMap");
  if (!isStreamingHeatMap) return;

  std::cout << "Heatmap update begin" << std::endl;
  heatMap.setTo(0);
  heatMapDirectionsMask.setTo(0);

  TopViewDrawer drawer(cv::Size(heatMap.cols, heatMap.rows));

  // float gridSize = 0.25; //in meters
  float gridSize = 0.1;  // in meters

  const auto& repr_particle = field_filter->getRepresentativeParticle();

  float halfFieldWidth =
      robocup_referee::Constants::field.border_strip_width_x + robocup_referee::Constants::field.field_length / 2;
  float halfFieldHeight =
      robocup_referee::Constants::field.border_strip_width_y + robocup_referee::Constants::field.field_width / 2;

  // Determinig size of a heatMapMat matrix
  // Y axis is inverted on field image
  FieldPosition p0(-halfFieldWidth, halfFieldHeight, 0.0);
  FieldPosition p1(halfFieldWidth, -halfFieldHeight, 0.0);
  FieldPosition ps(-halfFieldWidth + gridSize, halfFieldHeight - gridSize, 0.0);
  cv::Point2f pos_field_p0 = p0.getRobotPositionCV();
  cv::Point2f pos_field_p1 = p1.getRobotPositionCV();
  cv::Point2f pos_field_ps = ps.getRobotPositionCV();
  cv::Point2i pos_pix_p0 = drawer.getImgFromField(robocup_referee::Constants::field, pos_field_p0);
  cv::Point2i pos_pix_p1 = drawer.getImgFromField(robocup_referee::Constants::field, pos_field_p1);
  cv::Point2i pos_pix_ps = drawer.getImgFromField(robocup_referee::Constants::field, pos_field_ps);

  cv::Point2f pixstep = pos_pix_ps - pos_pix_p0;
  int heatMapMatCols = (pos_pix_p1.x - pos_pix_p0.x) / pixstep.x;
  int heatMapMatRows = (pos_pix_p1.y - pos_pix_p0.y) / pixstep.y;

  cv::Mat heatMapMat = cv::Mat(heatMapMatRows, heatMapMatCols, CV_32FC1);
  heatMapMat.setTo(255.0);

  for (float x = -halfFieldWidth; x < halfFieldWidth; x += gridSize) {
    for (float y = -halfFieldHeight; y < halfFieldHeight; y += gridSize) {
      FieldPosition pt(x, -y, 0.0);
      cv::Point2f pos_field = pt.getRobotPositionCV();
      cv::Point2i pix = drawer.getImgFromField(robocup_referee::Constants::field, pos_field) - pos_pix_p0;
      pix.x /= pixstep.x;
      pix.y /= pixstep.y;

      if (pix.x < 0) continue;
      if (pix.x > heatMapMat.cols - 1) continue;
      if (pix.y < 0) continue;
      if (pix.y > heatMapMat.rows - 1) continue;

      double bestAngle = 0.0;
      double bestAngleScore = 0.0;
      double angleScore;
      for (rhoban_unsorted::Observation<FieldPosition>* o : obs) {
        // for(double a=-180.0; a < 180.0; a+=5.0) { //sum all of possible orientations in heatmap mode
        double a = repr_particle.getOrientation().getSignedValue();
        {
          angleScore = 0.0;
          // FieldPosition p(x,y,a*M_PI/180.0);
          FieldPosition p(x + gridSize / 2, -(y + gridSize / 2), a);

          //[Sol] choose what to draw in HeatMap here

          //[Sol] Goal observations potential looks like rings around each goal post (in case of single post
          // observation) with particles uniformly distributed among this rings looking to post (with appropriate shift
          // estimated from camera angle) This is because feature itself (goal post) doesn't have orientation
          // information So to do localisation only by goal posts - knowledge of robot's current orientation from
          // gyro/odometry is essential. This prior knowledge of orientation will drop all particles with wrong
          // orientation from this rings and localisation will be ok
          /*
          GoalObservation * goalObs = dynamic_cast<GoalObservation *>(o);
          if (goalObs != nullptr) {
            double score = goalObs->potential(p);
            angleScore += score;
            addToHeatMap(heatMap, pix, pixstep, score);
          }
          */

          //[Sol] Arena corner observation is a feature with orientation estimated from detected field border angle.
          // So it's potential looks like a blob with particles heading appropriate direction
          // We have 4 corners on the arena so usually we will have 4 symmetrical clusters with angles shifted 90
          // degrees Arena border observation also have angle and it's potential will look line lines parallel to arena
          // borders with appropriate particle directions
          /*
          ArenaCornerObservation * acObs = dynamic_cast<ArenaCornerObservation *>(o);
          if (acObs != nullptr) {
            double score = acObs->potential(p);
            angleScore += score;
            //addToHeatMap(heatMap, pix, pixstep, score);
            //std::cout << score << " ";
          }
          */

          WhiteLinesCornerObservation* acObs = dynamic_cast<WhiteLinesCornerObservation*>(o);
          if (acObs != nullptr) {
            double score = acObs->potential(p);
            heatMapMat.at<float>(pix.y, pix.x) = heatMapMat.at<float>(pix.y, pix.x) * score;
            angleScore += score;

            // addToHeatMap(heatMap, pix, pixstep, score);
            // heatMapMat.at<float>(pix.y, pix.x) = heatMapMat.at<float>(pix.y, pix.x) * score;
            // std::cout << score << " ";
          }
          if (angleScore > bestAngleScore) {
            bestAngleScore = angleScore;
            bestAngle = a;
            // std::cout << "s="<<bestAngleScore<<"@"<<bestAngle*M_PI/180.0 << " ";
          }
        }
      }
      // std::cout << std::endl;

      // if(bestAngleScore>0) {
      // std::cout << bestAngle*M_PI/180.0 << " ";
      // Draw direction of best angle of potential in that point
      double dx = cos(bestAngle * M_PI / 180.0);
      double dy = -sin(bestAngle * M_PI / 180.0);
      double vecLength = 8.0;  // in pixels
      cv::Point2f delta(dx * vecLength, dy * vecLength);
      cv::Point2i deltai = cv::Point2i(delta.x, delta.y);
      double radius = 2;
      // cv::line(heatMapDirectionsMask, pixr+pixstep/2, pixr+pixstep/2+deltai, cv::Scalar(255), 1);
      // circle(heatMapDirectionsMask, pixr+pixstep/2, radius, cv::Scalar(255), -1);

      //}
    }
  }

  cv::Rect rectOfField = cv::Rect(pos_pix_p0.x, pos_pix_p0.y, pos_pix_p1.x - pos_pix_p0.x, pos_pix_p1.y - pos_pix_p0.y);
  // cv::resize(heatMapMat, heatMap(rectOfField), rectOfField.size(), 0 ,0, cv::INTER_NEAREST);
  cv::resize(heatMapMat, heatMap(rectOfField), rectOfField.size(), 0, 0, cv::INTER_LINEAR);

  // Drawing grid
  for (float x = -robocup_referee::Constants::field.field_length / 2 + 0.5;
       x < robocup_referee::Constants::field.field_length / 2 - 0.5; x += 1.0) {
    FieldPosition pg0(x, -halfFieldHeight, 0.0);
    FieldPosition pg1(x, halfFieldHeight, 0.0);
    cv::Point2f pos_field_pg0 = pg0.getRobotPositionCV();
    cv::Point2f pos_field_pg1 = pg1.getRobotPositionCV();
    cv::Point2f pix_pg0 = drawer.getImgFromField(robocup_referee::Constants::field, pos_field_pg0);
    cv::Point2f pix_pg1 = drawer.getImgFromField(robocup_referee::Constants::field, pos_field_pg1);
    cv::line(heatMap, pix_pg0, pix_pg1, cv::Vec3b(128, 128, 128), 1);
  }

  for (float y = -robocup_referee::Constants::field.field_width / 2;
       y < robocup_referee::Constants::field.field_width / 2; y += 1.0) {
    FieldPosition pg0(-halfFieldWidth, y, 0.0);
    FieldPosition pg1(halfFieldWidth, y, 0.0);
    cv::Point2f pos_field_pg0 = pg0.getRobotPositionCV();
    cv::Point2f pos_field_pg1 = pg1.getRobotPositionCV();
    cv::Point2f pix_pg0 = drawer.getImgFromField(robocup_referee::Constants::field, pos_field_pg0);
    cv::Point2f pix_pg1 = drawer.getImgFromField(robocup_referee::Constants::field, pos_field_pg1);
    cv::line(heatMap, pix_pg0, pix_pg1, cv::Vec3b(128, 128, 128), 1);
  }

  {
    FieldPosition pt = repr_particle;
    cv::Point2f pos_field = pt.getRobotPositionCV();
    cv::Point2i pix = drawer.getImgFromField(robocup_referee::Constants::field, pos_field) - pos_pix_p0;
    pix.x /= pixstep.x;
    pix.y /= pixstep.y;

    if ((pix.x >= 0) && (pix.x < heatMapMat.cols - 1) && (pix.y >= 0) && (pix.y < heatMapMat.rows - 1)) {
      double repr_final_pot = heatMapMat.at<float>(pix.y, pix.x) / 255.0;
      char s[255];
      sprintf(s, "Repr.particle final potentioal: %.2fs", repr_final_pot);
      int fontFace = cv::FONT_HERSHEY_PLAIN;
      double fontScale = 1.1;
      int thickness = 1;
      cv::putText(heatMap, std::string(s), cv::Point(0, 20), fontFace, fontScale, cv::Scalar::all(255), thickness,
                  CV_AA);
    }
  }

  std::cout << "end" << std::endl;
}

cv::Mat LocalisationBinding::getHeatMap(int width, int height) {
  if (cs == NULL) throw std::runtime_error("HeatMap is not ready");
  cv::Mat img(height, width, CV_8UC3);
  // Field::Field::drawField(img);
  field_filter->draw(img);  // 2019

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      cv::Vec3b c = img.at<cv::Vec3b>(y, x);
      // img.at<cv::Vec3b>(y,x) = cv::Vec3b(c[0], c[2], heatMap.at<float>(y,x));
      img.at<cv::Vec3b>(y, x) = cv::Vec3b(c[0], heatMap.at<float>(y, x), heatMap.at<float>(y, x));
    }
  }
  img.setTo(0, heatMapDirectionsMask);

  return img;
}

void LocalisationBinding::importFiltersResults() {
  filterMutex.lock();
  // Robot
  robot = field_filter->getRepresentativeParticle();
  robotQ = field_filter->getRepresentativeQuality();

  filterMutex.unlock();
}

/*cv::Mat LocalisationBinding::getTopView(int width, int height)
{
  filterMutex.lock();
  cv::Mat img(height, width, CV_8UC3);
  field_filter->draw(img);

  filterMutex.unlock();
  return img;
}*/

void LocalisationBinding::redrawTopViewBody(void) {
  // std::cout << "Topview redraw body" << std::endl;
  if (cs == NULL) throw std::runtime_error("TopView is not ready");

  filterMutex.lock();

  // double timer = (double)cv::getTickCount();

  cv::Mat img(740, 1040, CV_8UC3);
  field_filter->draw(img);
  // img.copyTo(extendedTopView(cv::Rect(0,100, 1040,740)));
  img.copyTo(extendedTopView(cv::Rect(0, 0, 1040, 740)));  // For header below topview

  // double elapsed =  1000.0 * (((double)cv::getTickCount() - timer) / cv::getTickFrequency());
  // fieldLogger.log("Topview body redraw time: %f ms", elapsed);

  filterMutex.unlock();
}

void LocalisationBinding::redrawTopViewHeader(void) {
  // std::cout << "Topview redraw header" << std::endl;
  // cv::Mat img = extendedTopView(cv::Rect(0,0, 1040,100));
  cv::Mat img = extendedTopView(cv::Rect(0, 740, 1040, 100));  // Draw  header below topview
  img.setTo(0);
  std::stringstream stream;
  double now_millis = getNowTS().getTimeMS();
  double elapsed = diffSec(currTS, getNowTS());
  int fontFace = cv::FONT_HERSHEY_PLAIN;
  double fontScale = 1.1;
  int thickness = 1;
  std::string text;
  char s[255];
  sprintf(s, "T since PF update: %.2fs", elapsed);
  cv::putText(img, std::string(s), cv::Point(0, 20), fontFace, fontScale, cv::Scalar::all(255), thickness, CV_AA);

  sprintf(s, "Consistency: %.2f", consistencyScore);
  cv::putText(img, std::string(s), cv::Point(0, 40), fontFace, fontScale, cv::Scalar::all(255), thickness, CV_AA);

  sprintf(s, "stepDelta: %+.2f", stepDeltaScore);
  cv::Scalar clr;
  if (stepDeltaScore > 0)
    clr = cv::Scalar(0, 255, 0);
  else
    clr = cv::Scalar(0, 0, 255);
  cv::putText(img, std::string(s), cv::Point(200, 40), fontFace, fontScale, clr, thickness, CV_AA);

  sprintf(s, "ObsSize: %d, good:%d, bad:%d, finalReprPot: %.2f, maxPot: %.2f", consistencyObsSize,
          consistencyGoodObsCount, consistencyBadObsCount, finalReprParticlePotential, maxPotential);
  cv::putText(img, std::string(s), cv::Point(0, 60), fontFace, fontScale, cv::Scalar::all(255), thickness, CV_AA);

  sprintf(s, "elapsedSinceUniformReset: %.2f/%.2f", elapsedSinceUniformReset, consistencyResetInterval);
  cv::putText(img, std::string(s), cv::Point(0, 80), fontFace, fontScale, cv::Scalar::all(255), thickness, CV_AA);

  // Drawing GLOBAL orientation vector using data from imu and gyroTareToOpGoal value
  double gyroYaw;  // in degrees
  // Code below is copy-paste from float Helpers::getGyroYaw() to correctly work in real mode and fake mode
  if (scheduler->isFakeMode()) {
    gyroYaw = rhoban::frameYaw(scheduler->getServices()->model->model.selfToWorld().rotation()) / M_PI * 180.0;
  } else {
    gyroYaw = scheduler->getManager()->dev<RhAL::GY85>("imu").getGyroYaw() / M_PI * 180.0;
  }

  rhoban_utils::Angle gyroYawToOpponentGoalAngle = rhoban_utils::Angle(gyroYawToOpponentGoal);
  rhoban_utils::Angle gyroYawAngle = rhoban_utils::Angle(gyroYaw);
  rhoban_utils::Angle yawAngle =
      gyroYawAngle - gyroYawToOpponentGoalAngle;  // Should give 0 when robot is facing opponent's goal
  double yaw = yawAngle.getSignedValue();

  double dx = cos(yawAngle);
  double dy = -sin(yawAngle);
  double vecLength = 40;  // in pixels
  cv::Point2f delta(dx * vecLength, dy * vecLength);

  circle(img, cv::Point2f(1040 - 50, 50), 45, cv::Scalar(255, 255, 255), 1, CV_AA);
  circle(img, cv::Point2f(1040 - 50, 50), 2, cv::Scalar(255, 255, 255), -1, CV_AA);
  cv::line(img, cv::Point2f(1040 - 50, 50), cv::Point2f(1040 - 50, 50) + delta, cv::Scalar(0, 255, 0), 2, CV_AA);
  // Print yaw angle above gauge
  sprintf(s, "%.1f", yaw);
  cv::putText(img, std::string(s), cv::Point(1040 - 50 - 30, 70), fontFace, fontScale, cv::Scalar::all(255), thickness,
              CV_AA);
  sprintf(s, "gyroYaw raw: %.1f", gyroYaw);
  cv::putText(img, std::string(s), cv::Point(1040 - 380, 20), fontFace, fontScale, cv::Scalar::all(255), thickness,
              CV_AA);

  if (gyroYawToOpponentGoalSetted)
    sprintf(s, "gyroYawToOpponnet: %.1f", gyroYawToOpponentGoal);
  else
    sprintf(s, "gyroYawToOpponnet: UNDEF");
  cv::putText(img, std::string(s), cv::Point(1040 - 380, 40), fontFace, fontScale, cv::Scalar::all(255), thickness,
              CV_AA);

  double gyroYawPot = getGyroYawPotential();
  if (gyroYawPot >= 0) {
    sprintf(s, "gyroYawObs pot: %.1f", gyroYawPot);
  } else {
    sprintf(s, "gyroYawObs pot: UNDEF");
  }
  cv::putText(img, std::string(s), cv::Point(1040 - 380, 60), fontFace, fontScale, cv::Scalar::all(255), thickness,
              CV_AA);
}

void LocalisationBinding::fieldReset(Localisation::FieldPF::ResetType type, float x, float y, float noise, float theta,
                                     float thetaNoise) {
  lastFieldReset = getNowTS();

  if (type == Localisation::FieldPF::ResetType::Custom) {
    auto rhioNode = &(RhIO::Root.child("/localisation/field/fieldPF"));
    rhioNode->setFloat("customX", x);
    rhioNode->setFloat("customY", y);
    rhioNode->setFloat("customNoise", noise);
    rhioNode->setFloat("customTheta", theta);
    rhioNode->setFloat("customThetaNoise", thetaNoise);
  }

  if (type == Localisation::FieldPF::Uniform) {
    lastUniformReset = lastFieldReset;
    consistencyScore = 0;
  } else if (type != Localisation::FieldPF::ResetType::Fall) {
    consistencyScore = 1;
  }
  field_filter->askForReset(type);
}

bool LocalisationBinding::refereeAllowsToPlay() {
  // On fake mode, always allow robot to play
  if (scheduler->isFakeMode() || !scheduler->getMove("robocup")->isRunning()) return true;

  RefereeService* referee = scheduler->getServices()->referee;
  bool allowedPhase = referee->isPlacingPhase() || referee->isFreezePhase();
  bool penalized = referee->isPenalized() && !referee->isServingPenalty();
  return referee->isPlaying() || (allowedPhase && !penalized);
}

// Get the best aviable potential on the filed _with current repr.particle orientation_
double LocalisationBinding::getMaxPotential(
    const std::vector<rhoban_unsorted::Observation<Localisation::FieldPosition>*>& obs) {
  float halfFieldWidth =
      robocup_referee::Constants::field.border_strip_width_x + robocup_referee::Constants::field.field_length / 2;
  float halfFieldHeight =
      robocup_referee::Constants::field.border_strip_width_y + robocup_referee::Constants::field.field_width / 2;

  const auto& repr_particle = field_filter->getRepresentativeParticle();
  double reprAngle = repr_particle.getOrientation().getSignedValue();

  double maxPot = 0;
  for (float x = -halfFieldWidth; x < halfFieldWidth;
       x +=
       0.2) {  // 0.1m granularity here will result in more percise consistency, but we need to shrink calculation time
    for (float y = -halfFieldHeight; y < halfFieldHeight; y += 0.2) {
      FieldPosition p(x, y, reprAngle);
      double pot = 1.0;
      for (rhoban_unsorted::Observation<FieldPosition>* o : obs) {
        pot *= o->potential(p);
      }
      if (pot > maxPot) maxPot = pot;
    }
  }

  return maxPot;
}

double LocalisationBinding::getGyroYawPotential() {
  if (gyroYawToOpponentGoalSetted) {
    double gyroYaw;  // in degrees
    if (scheduler->isFakeMode()) {
      gyroYaw = rhoban::frameYaw(scheduler->getServices()->model->model.selfToWorld().rotation()) / M_PI * 180.0;
    } else {
      gyroYaw = scheduler->getManager()->dev<RhAL::GY85>("imu").getGyroYaw() / M_PI * 180.0;
    }
    rhoban_utils::Angle gyroYawToOpponentGoalAngle = rhoban_utils::Angle(gyroYawToOpponentGoal);
    rhoban_utils::Angle gyroYawAngle = rhoban_utils::Angle(gyroYaw);
    rhoban_utils::Angle yawAngle = gyroYawAngle - gyroYawToOpponentGoalAngle;
    rhoban_unsorted::Observation<FieldPosition>* gyroYawObservation = new GyroYawObservation(yawAngle);
    const auto& repr_particle = field_filter->getRepresentativeParticle();
    double pot = gyroYawObservation->potential(repr_particle);
    return pot;
  } else {
    return -1;
  }
}

}  // namespace Vision

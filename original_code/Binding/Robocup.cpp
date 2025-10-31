#include "Binding/Robocup.hpp"
#include <iostream>
#include <unistd.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Localisation/Ball/BallStackFilter.hpp"
#include "Localisation/Ball/SpeedEstimator.hpp"

#include "Localisation/Robot/RobotFilter.hpp"

#include "scheduler/MoveScheduler.h"

#include "Filters/Features/FeaturesProvider.hpp"
#include "Filters/Features/TagsDetector.hpp"
#include "Filters/Source/SourceVideoProtobuf.hpp"

#include "CameraState/CameraState.hpp"
#include "Utils/Drawing.hpp"
#include "Utils/Interface.h"
#include "Utils/PtGreyExceptions.hpp"

#include "rhoban_geometry/point.h"

#include "rhoban_utils/timing/benchmark.h"

#include "RhIO.hpp"
#include "robocup_referee/constants.h"

#include "moves/Head.h"
#include <rhoban_utils/logging/logger.h>
#include <rhoban_utils/util.h>

#include "services/DecisionService.h"
#include "services/LocalisationService.h"
#include "services/ModelService.h"
#include "services/RefereeService.h"
//#include "services/ViveService.h"
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>
#include <unistd.h>

#include <vector>

#include <utility>
#include <algorithm>
#include "Localisation/Field/WhiteLinesCornerObservation.hpp"  //[Sol]
#include "Filters/Custom/WhiteLinesData.hpp"                   //[Sol]
#include "Binding/LocalisationBinding.hpp"                     //[Sol]
//#include "Filters/Custom/ObstacleMap.hpp"                    //[Sol]
#include "Filters/Custom/StereoImgProc.hpp"                    //[Sol]

#include <ros/ros.h>
#include "sensor_msgs/JointState.h"
#include "tf/transform_datatypes.h"
#include <tf_conversions/tf_eigen.h>

static rhoban_utils::Logger out("vision_robocup");

using namespace hl_monitoring;
using namespace Vision::Localisation;
using namespace rhoban_utils;
using namespace rhoban_geometry;

using namespace std;
using namespace std::chrono;

using namespace Vision::Utils;

using robocup_referee::Constants;
using Vision::Filters::TagsDetector;
using Vision::Utils::CameraState;
using Vision::Utils::ImageLogger;

/**
 * Capping image memory to 20 minutes at 40 fps
 */
static int max_images = 20 * 60 * 40;

namespace Vision {
Robocup::Robocup(MoveScheduler* scheduler)
    : Application(),
      manual_logger("manual_logs", false, max_images),
      moving_ball_logger("moving_ball_logs", false, 30 * 40)  // Less images memory for moving balls
      ,
      autologMovingBall(false),
      game_logger("game_logs", false, max_images),
      autolog_games(false),
      logBallExtraTime(2.0),
      writeBallStatus(false),
      _scheduler(scheduler),
      benchmark(false),
      benchmarkDetail(0),
      cs(new CameraState(scheduler)),
      activeSource(false),
      clearRememberObservations(false),
      detectedFeatures(new Field::POICollection()),
      detectedBalls(new std::vector<cv::Point3f>()),
      detectedBallsSelf(new std::vector<cv::Point2f>())  //[Sol] debug
      ,
      detectedRobots(new std::vector<cv::Point3f>()),
      wasHandled(false),
      wasFallen(false),
      ignoreOutOfFieldBalls(true),
      treatmentDelay(0),
      accumulateApproachVerbose(false),
      accumulateApproachVerboseOld(false),
      ballPosOnApproachAccumulatedImg(cv::Point2f(0, 0)),
      ballAngleOnApproachAccumulatedImg(0),
      approachImgSavedImageNumber(0),
      approachImgSavedApproachNumber(0) {
  ballStackFilter = new BallStackFilter(cs);
  robotFilter = new RobotFilter(cs);
  ballSpeedEstimator = new SpeedEstimator();
  initObservationTypes();

  for (std::string obs : observationTypes) {
    rememberObservations[obs] = std::vector<std::pair<cv::Point2f, float>>();
  }

  out.log("Starting Robocup Pipeline");
  initImageHandlers();
  loadFile();
  _doRun = true;
  Filter::GPU_ON = gpuOn;
  if (pathToLog != "") {
    // The low level info will come from a log
    setLogMode(pathToLog);
  }
  if (viveLogPath != "") {
    setViveLog(viveLogPath);
  }
  kmc.loadFile();
  taggedKickName = "classic";

  pipeline.setCameraState(cs);
  pipeline.setCameraState(cs, true); //Setting default cameraState for stereo thread


  // Hack to be able to apply/remove CameraCalibration when replaying Log
  try {
    try {
      // Filters::SourceVideoProtobuf& source_pb = dynamic_cast<Filters::SourceVideoProtobuf&>(pipeline.get("human"));
      // //No wideangle version
      Filters::SourceVideoProtobuf& source_pb =
          dynamic_cast<Filters::SourceVideoProtobuf&>(pipeline.get("sourceRaw"));  // Wideangle version
      source_pb.setScheduler(scheduler);
    } catch (const std::bad_cast& e) {
      // This warning is OK for first frames during live vision init, but will generate SIGSEGV (absent moveSheduler
      // ptr) in fake mode
      std::cout << "Robocup.cpp: cannot cast SourceVideoProtobuf's input filter from pipeline" << std::endl;
    }

    try {
      Filters::SourceVideoProtobuf& source_pb_stereo =
          dynamic_cast<Filters::SourceVideoProtobuf&>(pipeline.get("sourceStereo"));  // Stereo version
      source_pb_stereo.setScheduler(scheduler);
    } catch (const std::bad_cast& e) {
      // This warning is OK for first frames during live vision init, but will generate SIGSEGV (absent moveSheduler
      // ptr) in fake mode
      std::cout << "Robocup.cpp: cannot cast SourceVideoProtobuf2's input filter from pipeline" << std::endl;
    }
  } catch (const std::runtime_error& e) {
    // Do nothing
  }

  scheduler->getServices()->localisation->setRobocup(this);
  _runThread = new std::thread(std::bind(&Robocup::run, this));
}

Robocup::~Robocup() {
  delete ballStackFilter;
  _doRun = false;
  if (_runThread != NULL) {
    _runThread->join();
    delete _runThread;
  }
}

void Robocup::startLogging(unsigned int timeMS, const std::string& logDir) {
  // If logDir is empty a name session is generated automatically in manual_logger
  logMutex.lock();
  manual_logger.initSession(*cs, pipeline.isFilterPresent("sourceRaw2"), logDir);
  startLoggingLowLevel(manual_logger.getSessionPath() + "/lowLevel.log");
  endLog = getNowTS().addMS(timeMS);
  logMutex.unlock();
}

void Robocup::endLogging() {
  logMutex.lock();
  // Telling the low level to stop logging and to dump the info
  stopLoggingLowLevel(manual_logger.getSessionPath() + "/lowLevel.log");
  manual_logger.endSession();
  // TODO: examine if logMutex can be closed earlier
  logMutex.unlock();
}

cv::Mat Robocup::getImg(const std::string& name, int wishedWidth, int wishedHeight, bool gray) {
  cv::Mat original, scaled, final;
  if (name == "Tagged") {
    original = getTaggedImg();
  } else {
    try {
      original = pipeline.get(name).getImg()->clone();
    } catch (const std::out_of_range& o) {
      throw std::runtime_error("Image not found : '" + name + "'");
    }
  }
  cv::resize(original, scaled, cv::Size(wishedWidth, wishedHeight));
  if (gray) {
    cv::cvtColor(scaled, final, CV_RGB2GRAY);
  } else {
    final = scaled;
  }
  return final;
}

Json::Value Robocup::toJson() const {
  // Writing stream
  Json::Value v = Application::toJson();
  v["viveLogPath"] = viveLogPath;
  v["benchmark"] = benchmark;
  v["benchmarkDetail"] = benchmarkDetail;
  v["autologMovingBall"] = autologMovingBall;
  v["autologGames"] = autolog_games;
  v["logBallExtraTime"] = logBallExtraTime;
  v["writeBallStatus"] = writeBallStatus;
  v["ignoreOutOfFieldBalls"] = ignoreOutOfFieldBalls;
  v["feature_providers"] = vector2Json(featureProviders);
  for (const SpecialImageHandler& sih : imageHandlers) {
    v[sih.name] = sih.display;
  }
  return v;
}

void Robocup::fromJson(const Json::Value& v, const std::string& dir_name) {
  Application::fromJson(v, dir_name);
  rhoban_utils::tryRead(v, "viveLogPath", &viveLogPath);
  rhoban_utils::tryRead(v, "benchmark", &benchmark);
  rhoban_utils::tryRead(v, "benchmarkDetail", &benchmarkDetail);
  rhoban_utils::tryRead(v, "autologMovingBall", &autologMovingBall);
  rhoban_utils::tryRead(v, "autologGames", &autolog_games);
  rhoban_utils::tryRead(v, "logBallExtraTime", &logBallExtraTime);
  rhoban_utils::tryRead(v, "writeBallStatus", &writeBallStatus);
  rhoban_utils::tryRead(v, "ignoreOutOfFieldBalls", &ignoreOutOfFieldBalls);
  rhoban_utils::tryReadVector<std::string>(v, "featureProviders", &featureProviders);
  for (SpecialImageHandler& sih : imageHandlers) {
    rhoban_utils::tryRead(v, sih.name, &sih.display);
    if (embedded) {
      std::cout << "Disabling " << sih.name << std::endl;
      sih.display = false;
    }
  }
}

void Robocup::init() {
  Application::init();
  lastTS = ::rhoban_utils::TimeStamp::fromMS(0);

  initRhIO();
}

void Robocup::initImageHandlers() {
  imageHandlers.push_back(SpecialImageHandler(
      "TaggedImg", 640, 480, [this](int width, int height) { return this->getTaggedImg(width, height); }));
  imageHandlers.push_back(SpecialImageHandler(
      "RadarImg", 640, 480, [this](int width, int height) { return this->getRadarImg(width, height); }));
  imageHandlers.push_back(SpecialImageHandler("ApproachImg", 640, 480, [this](int width, int height) {
    return this->getApproachImg(/*width, height*/);
  }));  // ApproachImg will be the same size as birdview
}

void Robocup::initRhIO() {
  // If a command has already been created do not pass here again
  if (RhIO::Root.commandExist("Vision/logLocal")) {
    return;
  }
  RhIO::Root.newStr("/Vision/cameraStatus")->defaultValue("");
  RhIO::Root.newFloat("/Vision/treatmentDelay")
      ->defaultValue(-1)
      ->comment("Time between image acquisition and result publication [ms]");
  RhIO::Root.newFloat("/Vision/lastUpdate")->defaultValue(-1)->comment("Time since last update [ms]");
  // Init interface with RhIO
  if (isFakeMode()) {  /// Highgui is not available on robot
    RhIO::Root.newCommand("Vision/showFilters", "Display the given filters",
                          [this](const std::vector<std::string>& args) -> std::string {
                            if (args.size() < 1) {
                              throw std::runtime_error("Usage: showFilters <name1> <name2> ...");
                            }
                            for (const std::string& name : args) {
                              try {
                                pipeline.get(name).display = true;
                              } catch (const std::out_of_range& exc) {
                                throw std::runtime_error("Filter " + name + " is not found in pipeline");
                              }
                            }
                            return "Filters are now displayed";
                          });
    RhIO::Root.newCommand("Vision/hideFilters", "Hide the given filters",
                          [this](const std::vector<std::string>& args) -> std::string {
                            if (args.size() < 1) {
                              throw std::runtime_error("Usage: hideFilters <name1> <name2> ...");
                            }
                            for (const std::string& name : args) {
                              try {
                                pipeline.get(name).display = false;
                                cv::destroyWindow(name);
                              } catch (const std::out_of_range& exc) {
                                throw std::runtime_error("Filter " + name + " is not found in pipeline");
                              }
                            }
                            return "Filters are now hidden";
                          });
  }
  RhIO::Root.newCommand("Vision/logLocal",
                        "Starts logging for a specified duration. Images are "
                        "saved on board for now.",
                        [this](const std::vector<std::string>& args) -> std::string {
                          if (args.size() < 1) {
                            throw std::runtime_error("Usage: logLocal <duration[s]> <opt:log_dir>");
                          }
                          double duration = std::stof(args[0]);
                          std::string logDir("");
                          if (args.size() >= 2) {
                            logDir = args[1];
                          }
                          this->startLogging((unsigned int)(duration * 1000), logDir);
                          return "";
                        });
  RhIO::Root.newBool("/Vision/autologMovingBall")
      ->defaultValue(autologMovingBall)
      ->comment("If enabled, start writing logs each time the ball is considered as moving");

  RhIO::Root.newFloat("/Vision/cameraYaw")->defaultValue(camera_yaw)->comment("camera yaw");

  RhIO::Root.newFloat("/Vision/logBallExtraTime")
      ->defaultValue(logBallExtraTime)
      ->comment("Extra duration of log once ball stopped being flagged as moving [s]");

  RhIO::Root.newBool("/Vision/autologGames")
      ->defaultValue(autolog_games)
      ->comment("If enabled, write logs while game is playing");
  RhIO::Root.newBool("/Vision/benchmark")->defaultValue(benchmark)->comment("Is logging activated ?");
  RhIO::Root.newInt("/Vision/benchmarkDetail")->defaultValue(benchmarkDetail)->comment("Depth of print for benchmark");
  RhIO::Root.newFloat("/Vision/motorDelay")
      ->defaultValue(CameraState::motor_delay)
      ->maximum(20)
      ->minimum(-20)
      ->comment("Monotonic TS = schedulerTS - motor_delay");

  // Monitoring special images
  for (const SpecialImageHandler& sih : imageHandlers) {
    std::string prefix = "Vision/" + sih.name;
    RhIO::Root.newFloat(prefix + "_scale")->defaultValue(1.0)->comment("");
    RhIO::Root.newFrame(prefix, "");
  }
  RhIO::Root.newStr("/Vision/taggedKickName")->defaultValue("classic");

  RhIO::Root.newBool("/Vision/accumulateApproachVerbose")
      ->defaultValue(accumulateApproachVerbose)
      ->comment("Starts to accumulate approachVerboseImage when set to true. Autoclean on finished approach");

  ballStackFilter->bindToRhIO("ballStack", "ballStack");

  robotFilter->bindToRhIO("robotFilter", "robotFilter");
}

void Robocup::initObservationTypes() {
  observationTypes = {"ball", "robot", "tag"};
  for (Field::POIType type : Field::getPOITypeValues()) {
    observationTypes.push_back(Field::poiType2String(type));
  }
}

void Robocup::finish() {}

void Robocup::step() {
  if (clearRememberObservations) {
    for (auto& entry : rememberObservations) {
      entry.second.clear();
    }
    clearRememberObservations = false;
  }

  // Sometimes vision is useless and even bug prone, in this case, cancel the
  // step
  DecisionService* decision = _scheduler->getServices()->decision;
  bool handled = decision->handled;
  bool fallen = decision->isFallen;
  if (embedded && (handled || fallen)) {
    //publishToRhIO();
    std::ostringstream oss;
    // Updating handled/fallen and sending message
    if (!wasHandled && handled) {
      out.log("Disabling vision (handled)");
      wasHandled = true;
    }
    if (wasHandled && !handled) {
      out.log("Robot is not handled anymore");
      wasHandled = false;
    }
    if (!wasFallen && fallen) {
      wasFallen = true;
      out.log("Disabling vision (fallen)");
    }
    if (wasFallen && !fallen) {
      out.log("Robot is not fallen anymore");
      wasFallen = false;
    }
    // Sleeping and waiting next step
    //int ms_sleep = 10;
    //usleep(ms_sleep * 1000);
    //return;
  }
  /*if (wasHandled || wasFallen) {
    out.log("Starting to step vision again");
    wasHandled = false;
    wasFallen = false;
  }*/

  // If the camera is unplugged or doesn't respond for too long, we should avoid
  // keeping the filters data
  {
    double timeSinceLastFrame = diffSec(lastTS, getNowTS());
    if (timeSinceLastFrame > 5) {
      out.warning("no frame for %f, reseting the ball filter", timeSinceLastFrame);

      // Resetting the ball stack filter
      ballStackFilter->clear();

      // Telling the localisation
      _scheduler->getServices()->localisation->setNoBall();
    }
  }

  // Making sure the image delay is given to the pipeline
  importFromRhIO();
  Benchmark::open("Vision + Localisation");

  Benchmark::open("Waiting for global mutex");
  globalMutex.lock();
  Benchmark::close("Waiting for global mutex");

  Benchmark::open("Pipeline");
  try {
    // Signal pipeline shoud the benchmark debug print should be used or not
    pipeline.benchmarkFromRobocup = benchmark;
    pipeline.benchmarkDetailFromRobocup = benchmarkDetail;

    double startTimer = (double)cv::getTickCount();

    Application::step();

    double elapsed_real_ms = 1000.0 * (((double)cv::getTickCount() - startTimer) / cv::getTickFrequency());
    pipelineFps = 1000.0 / elapsed_real_ms;
    if (pipelineFpsFiltered == 0) pipelineFpsFiltered = pipelineFps;
    pipelineFpsFiltered = pipelineFpsFiltered * 0.9 + pipelineFps * 0.1;

    moveSchedulerFps = 1000.0 / _scheduler->getDurationCycle();
    if (moveSchedulerFpsFiltered == 0) moveSchedulerFpsFiltered = moveSchedulerFps;
    moveSchedulerFpsFiltered = moveSchedulerFpsFiltered * 0.9 + moveSchedulerFps * 0.1;

    bool print_fps = true;
    if (print_fps) {
      char s[50];
      sprintf(s, "pipeline FPS: %04.1f \t scheduler FPS: %04.1f", pipelineFpsFiltered, moveSchedulerFpsFiltered);
      std::cout << std::string(s) << std::endl;
    }

    activeSource = true;
    // If Vision application has finished, ask for scheduler to shut down
    if (!isActive()) {
      out.log("Vision exiting, asking to scheduler to shut down");
      _scheduler->askQuit();
    }
    ros::spinOnce();
  } catch (const PtGreyException& exc) {
    globalMutex.unlock();
    Benchmark::close("Pipeline");
    Benchmark::close("Vision + Localisation", benchmark, benchmarkDetail);
    activeSource = false;
    out.warning("Failed vision step: '%s'", exc.what());
    publishToRhIO();
    int sleep_time_ms = 100;
    usleep(sleep_time_ms * 1000);
    return;
  } catch (const PtGreyConnectionException& exc) {
    globalMutex.unlock();
    Benchmark::close("Pipeline");
    Benchmark::close("Vision + Localisation", benchmark, benchmarkDetail);
    activeSource = false;
    out.warning("Failed to connect to camera: '%s'", exc.what());
    publishToRhIO();
    int sleep_time_ms = 500;
    usleep(sleep_time_ms * 1000);
    return;
  }
  Benchmark::close("Pipeline");

  if (embedded && (handled || fallen)) {
    globalMutex.unlock();    
    displaySpecialImagehandlers(); //Stream TaggedImg/Radarimg/etc on live robot, and filters with display=true in fake
    publishToRhIO();
    return;
  }

  Benchmark::open("readPipeline");
  getUpdatedCameraStateFromPipeline();
  readPipeline();
  Benchmark::close("readPipeline");

  Benchmark::open("loggingStep");
  loggingStep();
  Benchmark::close("loggingStep");

  Benchmark::open("BallInformations");
  updateBallInformations();
  Benchmark::close("BallInformations");

  Benchmark::open("RobotInformations");
  updateRobotInformations();
  Benchmark::close("RobotInformations");

  
  

  // if (pipeline.isFilterPresent("obstacleMap")) {
  //   Benchmark::open("StereoObstaclesInformations");
  //   try {
  //     Filters::ObstacleMap& obstacleMap = dynamic_cast<Filters::ObstacleMap&>(pipeline.get("obstacleMap"));

  //     std::vector<Filters::StereoObstacle> detectedObstaclesSelf = obstacleMap.getDetectedObstacles();
  //     /*std::vector<Filters::StereoObstacle> detectedObstaclesWorld;
  //     for(auto obstacle: detectedObstaclesSelf) {
  //       std::vector<Eigen::Vector2d> points_in_self = obstacle.hull_points_in_self;
  //       std::vector<Eigen::Vector3d> points_in_world;
  //       for(size_t i=0;i< points_in_self.size();i++) {
  //         points_in_world.push_back( cs->getWorldFromSelf(Eigen::Vector3d(points_in_self[i].x(), points_in_self[i].y(),
  //     0)) );
  //       }
  //     }  */
  //     // Broadcast information to localisation service
  //     LocalisationService* loc = _scheduler->getServices()->localisation;
  //     // loc->setOpponentsWorld(robot_candidates);
  //     loc->setStereoObstaclesSelf(detectedObstaclesSelf);


  //   } catch (const std::runtime_error& e) {
  //     out.warning("%s, cannot cast StereoImgProc from pipline for obstacle transfer to localisation", e.what());
  //   }
  //   Benchmark::close("StereoObstaclesInformations");
  // }

  Benchmark::open("Tagging & Display");

  globalMutex.unlock();

  //Stream TaggedImg/Radarimg/etc on live robot, and filters with display=true in fake
  displaySpecialImagehandlers();

  Benchmark::open("Waiting for global mutex");
  globalMutex.lock();
  Benchmark::close("Waiting for global mutex");

  treatmentDelay = diffMs(sourceTS, TimeStamp::now());
  publishToRhIO();

  Benchmark::close("Tagging & Display");

  globalMutex.unlock();

  Benchmark::close("Vision + Localisation", benchmark, benchmarkDetail);

  // Set the log timestamp during fake mode
  if (isFakeMode()) {
    double ts = pipeline.getCameraState()->getTimeStampDouble();
    _scheduler->getServices()->model->setReplayTimestamp(ts);
  }
}

void Robocup::importFromRhIO() {
  autologMovingBall = RhIO::Root.getValueBool("/Vision/autologMovingBall").value;
  autolog_games = RhIO::Root.getValueBool("/Vision/autologGames").value;
  logBallExtraTime = RhIO::Root.getValueFloat("/Vision/logBallExtraTime").value;
  benchmark = RhIO::Root.getValueBool("/Vision/benchmark").value;
  benchmarkDetail = RhIO::Root.getValueInt("/Vision/benchmarkDetail").value;
  CameraState::motor_delay = RhIO::Root.getValueFloat("/Vision/motorDelay").value;
  // Import size update for images
  for (SpecialImageHandler& sih : imageHandlers) {
    std::string prefix = "Vision/" + sih.name;
    sih.scale = RhIO::Root.getValueFloat(prefix + "_scale").value;
  }
  taggedKickName = RhIO::Root.getValueStr("/Vision/taggedKickName").value;
  accumulateApproachVerbose = RhIO::Root.getValueBool("/Vision/accumulateApproachVerbose").value;
}

void Robocup::publishToRhIO() {
  RhIO::Root.setFloat("/Vision/treatmentDelay", treatmentDelay);
  RhIO::Root.setFloat("/Vision/lastUpdate", diffMs(lastTS, getNowTS()));
  std::string cameraStatus = getCameraStatus();
  RhIO::Root.setStr("/Vision/cameraStatus", cameraStatus);
  RhIO::Root.setBool("/Vision/accumulateApproachVerbose", accumulateApproachVerbose);
}

std::string Robocup::getCameraStatus() const {
  if (!activeSource) {
    return "Connection lost";
  }
  DecisionService* decision = _scheduler->getServices()->decision;
  if (decision->handled || decision->isFallen) {
    return "Inactive (handled or fallen)";
  }
  return "Active";
}

void Robocup::readPipeline() {
  featuresMutex.lock();
  // Ball and robots are cleared after every step (used internally)
  detectedBalls->clear();
  detectedBallsSelf->clear();
  detectedRobots->clear();
  for (const auto& provider_name : featureProviders) {
    try {
      const Filters::FeaturesProvider& provider =
          dynamic_cast<const Filters::FeaturesProvider&>(pipeline.get(provider_name));
      // Balls import
      std::vector<cv::Point2f> balls_in_img = provider.getBalls();
      for (const cv::Point2f& ball_in_img : balls_in_img) {
        // Eigen::Vector3d ball_pos = cs->ballInWorldFromPixel(ball_in_img);
        // Eigen::Vector3d ball_pos = cs->ballInWorldFromPixelWideangle(ball_in_img);
        Eigen::Vector3d ball_pos = cs->ballInWorldFromPixel(ball_in_img, CAMERA_WIDE_FULL);
        detectedBalls->push_back(eigen2CV(ball_pos));

        cv::Point2f ball_pos_self = cs->robotPosFromImg(ball_in_img.x, ball_in_img.y, CAMERA_WIDE_FULL);
        detectedBallsSelf->push_back(ball_pos_self);
      }
      // POI update
      std::map<Field::POIType, std::vector<cv::Point2f>> pois_in_img = provider.getPOIs();
      for (const auto& entry : pois_in_img) {
        Field::POIType poi_type = entry.first;
        for (const cv::Point2f& feature_pos_in_img : entry.second) {
          cv::Point2f world_pos = cs->worldPosFromImg(feature_pos_in_img.x, feature_pos_in_img.y);
          detectedFeatures->operator[](poi_type).push_back(cv::Point3f(world_pos.x, world_pos.y, 0));
        } 
      }
      // Robot import
       std::vector<cv::Point2f> robots_in_img = provider.getRobots();  // TODO: add robot info (color)
       for (const cv::Point2f& robot_in_img : robots_in_img) {
      //   // cv::Point2f world_pos = cs->worldPosFromImg(robot_in_img.x, robot_in_img.y);
      //   // cv::Point2f world_pos = cs->worldPosFromImgWideangle(robot_in_img.x, robot_in_img.y); //[Sol] temp test
         cv::Point2f world_pos = cs->worldPosFromImg(robot_in_img.x, robot_in_img.y, CAMERA_WIDE_FULL); // wtf CAMERA_WIDE_FULL
         detectedRobots->push_back(cv::Point3f(world_pos.x, world_pos.y, 0));
             //std::cout << "detectedRobots.size()=" << detectedRobots->size() << std::endl;
            //for(size_t i=0; i < detectedRobots->size(); i++) {
            //  std::cout << "   robot[" << i <<"] pos_world=" << (*detectedRobots)[i] << std::endl;
          //}
       }
    } catch (const std::bad_cast& e) {
      out.error("%s: Failed to import features, check pipeline. Exception: %s", DEBUG_INFO.c_str(), e.what());
    } catch (const std::runtime_error& exc) {
      out.error("%s: Failed to import features, runtime_error. Exception: %s", DEBUG_INFO.c_str(), exc.what());
    }
  }

  // [Sol] Adding detected stereo vision obstacles as opponent robot candidates, they will be filtered/merged/etc by robotFilter later
  //std::cout << "detectedRobots:" << std::endl;
  if (pipeline.isFilterPresent("stereoImgProc")) {
    Benchmark::open("StereoObstaclesInformations");
    try {
      Filters::StereoImgProc& stereoImgProc = dynamic_cast<Filters::StereoImgProc&>(pipeline.get("stereoImgProc"));

      std::vector<Filters::StereoCircularObstacle> detectedObstaclesSelf = stereoImgProc.getDetectedObstacles();
      for(auto obstacle: detectedObstaclesSelf) {
        detectedRobots->push_back(cv::Point3f(obstacle.pos_in_world.x(), obstacle.pos_in_world.y(), 0)); 
      } 
    } catch (const std::runtime_error& e) {
      out.warning("%s, cannot cast StereoImgProc from pipline for obstacle transfer to localisation", e.what());
    }
    //std::cout << "detectedRobots.size()=" << detectedRobots->size() << std::endl;
    //for(size_t i=0; i < detectedRobots->size(); i++) {
    //  std::cout << "   robot[" << i <<"] pos_world=" << (*detectedRobots)[i] << std::endl;
    //}
    Benchmark::close("StereoObstaclesInformations");
  }  

  featuresMutex.unlock();

  //[Sol]
  whiteLinesMutex.lock();
  if (pipeline.isFilterPresent("whiteLinesBirdviewDetector")) {
    Vision::Filter& whiteLinesBirdview_f = pipeline.get("whiteLinesBirdviewDetector");
    whitelines_data_single_frame.clear();
    try {
      const Vision::Filters::WhiteLinesBirdview& whiteLinesBirdview =
          dynamic_cast<const Vision::Filters::WhiteLinesBirdview&>(whiteLinesBirdview_f);
      std::vector<Vision::Filters::WhiteLinesData> data_vector = whiteLinesBirdview.loc_data_vector;
      for (int i = 0; i < data_vector.size(); i++) {
        whitelines_data_accumulated.push_back(data_vector[i]);
        whitelines_data_single_frame.push_back(data_vector[i]);
      }
    } catch (const std::bad_cast& e) {
      // out.log("Failed to cast filter '%s' in whiteLines", name.c_str());
      out.log("Failed to cast filter in WhiteLinesBirdview");
    }
  }
  whiteLinesMutex.unlock();

  // Tags (temporarily disabled, to reactivate, require to add a 'tagProvider' similar to featureProviders
  //  for (const std::string& tagProviderName : tagProviders)
  //  {
  //    Vision::Filter& tagFilter = pipeline.get(tagProviderName);
  //    cv::Size size = pipeline.get(tagProviderName).getImg()->size();
  //    tagsMutex.lock();
  //    detectedTimestamp = pipeline.getTimestamp().getTimeMS() / 1000.0;
  //    try
  //    {
  //      const TagsDetector& tagProvider = dynamic_cast<const TagsDetector&>(tagFilter);
  //      const std::vector<TagsDetector::Marker>& new_tags = tagProvider.getDetectedMarkers();
  //      for (const TagsDetector::Marker& marker : new_tags)
  //      {
  //        Eigen::Vector3d pos_camera;
  //        cv::cv2eigen(marker.tvec, pos_camera);
  //        Eigen::Vector3d marker_pos_in_world = cs->getWorldPosFromCamera(pos_camera);
  //
  //        // Adding Marker to detectedTagsb
  //        detectedTagsIndices.push_back(marker.id);
  //        detectedTagsPositions.push_back(marker_pos_in_world);
  //
  //        // Calculating the center of the tags on the image (x, y)
  //        // TODO, if the barycenter is not good enough, do better (crossing the
  //        // diags?)
  //        Eigen::Vector2d avg_in_img(0, 0);
  //        for (unsigned int i = 0; i < marker.corners.size(); i++)
  //        {
  //          avg_in_img += Eigen::Vector2d(marker.corners[i].x, marker.corners[i].y);
  //        }
  //        avg_in_img /= 4.0;
  //        // Rescaling to cameraModel image
  //        avg_in_img(0) *= cs->getCameraModel().getImgWidth() / size.width;
  //        avg_in_img(1) *= cs->getCameraModel().getImgHeight() / size.height;
  //        // Undistort position
  //        cv::Point2f avg_in_corrected;
  //        avg_in_corrected = cs->getCameraModel().toCorrectedImg(eigen2CV(avg_in_img));
  //        // Using a pair instead than a cv::Point so the structure is usable even
  //        // without opencv (will be read by the low level)
  //        std::pair<float, float> pair_in_img(avg_in_img(0), avg_in_img(1));
  //        std::pair<float, float> pair_undistorded(avg_in_corrected.x, avg_in_corrected.y);
  //        detectedTagsCenters.push_back(pair_in_img);
  //        detectedTagsCentersUndistort.push_back(pair_undistorded);
  //      }
  //    }
  //    catch (const std::bad_cast& exc)
  //    {
  //      tagsMutex.unlock();
  //      throw std::runtime_error("Invalid type for filter 'TagsDetector'");
  //    }
  //    catch (...)
  //    {
  //      tagsMutex.unlock();
  //      throw;
  //    }
  //    tagsMutex.unlock();
  //  }
}

void Robocup::getUpdatedCameraStateFromPipeline() {
  csMutex.lock();

  // Backup values
  lastTS = sourceTS;

  // EXPERIMENTAL:
  //
  // modification linked to the possibility that 'Source' filter provides the
  // cameraState (from SourceVideoProtobuf)
  cs = pipeline.getCameraState();
  ballStackFilter->updateCS(cs);
  robotFilter->updateCS(cs);

  sourceTS = cs->getTimeStamp();

  // TODO: identify if this part is only debug
  if (!isFakeMode()) {
    double timeSinceLastFrame = diffSec(lastTS, sourceTS);
    if (timeSinceLastFrame > 2 || timeSinceLastFrame < 0) {
      out.warning("Suspicious elapsed time: %f [s]", timeSinceLastFrame);
    }
  }

  csMutex.unlock();
}

void Robocup::loggingStep() {
  TimeStamp now = getNowTS();

  DecisionService* decision = _scheduler->getServices()->decision;
  RefereeService* referee = _scheduler->getServices()->referee;

  ImageLogger::Entry entry;
  // Capture src_filter and throws a std::runtime_error if required
  // const Filter& src_filter = pipeline.get("human"); //[Sol] old version
  const Filter& src_filter_l = pipeline.get("sourceRaw");  //[Sol] wide-angle stereo version
  entry.img_l = *(src_filter_l.getImg());

  if (pipeline.isFilterPresent("sourceRaw2")) {
    const Filter& src_filter_r = pipeline.get("sourceRaw2");  //[Sol] wide-angle stereo version
    entry.img_r = *(src_filter_r.getImg());
  } else {
    entry.img_r = cv::Mat();  // Mat with zero width/height
  }

  entry.time_stamp = (uint64_t)(pipeline.getTimestamp().getTimeSec() * std::pow(10, 6));
  entry.cs = *cs;

  logMutex.lock();
  // Handling manual logs
  bool dumpManualLogs = false;
  if (manual_logger.isActive()) {
    if (endLog < now) {  // If time is elapsed: close log
      dumpManualLogs = true;
    } else {  // If there is still time left, add entry
      try {
        manual_logger.pushEntry(entry, pipeline.isFilterPresent("sourceRaw2"));
      } catch (const ImageLogger::SizeLimitException& exc) {
        out.warning("Automatically stopping manual log because size limit was reached");
        dumpManualLogs = true;
      }
    }
  }
  // Handling moving ball logs
  if (decision->isBallMoving) {
    lastBallMoving = now;
  }
  double elapsedSinceBallMoving = diffSec(lastBallMoving, now);
  // Status of autoLog
  bool autoLogActive = moving_ball_logger.isActive();
  bool startAutoLog = autologMovingBall && !autoLogActive && decision->hasMateKickedRecently;
  bool stopAutoLog = autoLogActive && !(elapsedSinceBallMoving < logBallExtraTime || decision->hasMateKickedRecently);
  bool useAutoLogEntry = startAutoLog || (autoLogActive && !stopAutoLog);
  // Starting autoLog
  if (startAutoLog) {
    moving_ball_logger.initSession(*cs, pipeline.isFilterPresent("sourceRaw2"));
    out.log("Starting a session at '%s'", moving_ball_logger.getSessionPath().c_str());
    std::string lowLevelPath = moving_ball_logger.getSessionPath() + "/lowLevel.log";
    startLoggingLowLevel(lowLevelPath);
  }
  // Trying to log entry (can fail is maxSize is reached)
  if (useAutoLogEntry) {
    try {
      moving_ball_logger.pushEntry(entry, pipeline.isFilterPresent("sourceRaw2"));
    } catch (const ImageLogger::SizeLimitException& exc) {
      stopAutoLog = true;
    }
  }

  // Status of game_logs
  bool is_log_allowed = !referee->isInitialPhase() && !referee->isFinishedPhase() && !referee->isPenalized();
  bool gameLogActive = game_logger.isActive();
  bool useGameLogEntry = autolog_games && is_log_allowed;
  bool startGameLog = !gameLogActive && useGameLogEntry;
  bool stopGameLog = gameLogActive && !useGameLogEntry;
  if (startGameLog) {
    game_logger.initSession(*cs, pipeline.isFilterPresent("sourceRaw2"));
    out.log("Starting a session at '%s'", game_logger.getSessionPath().c_str());
    std::string lowLevelPath = game_logger.getSessionPath() + "/lowLevel.log";
    startLoggingLowLevel(lowLevelPath);
  }
  if (useGameLogEntry) {
    try {
      game_logger.pushEntry(entry, pipeline.isFilterPresent("sourceRaw2"));
    } catch (const ImageLogger::SizeLimitException& exc) {
      stopGameLog = true;
    }
  }
  logMutex.unlock();

  // Writing logs is delayed until logMutex has been unlocked to avoid
  // unnecessary lock of ressources
  if (dumpManualLogs) {
    _scheduler->stopMove("head", 0.5);
    endLogging();
  }
  if (stopAutoLog) {
    std::string lowLevelPath = moving_ball_logger.getSessionPath() + "/lowLevel.log";
    stopLoggingLowLevel(lowLevelPath);
    moving_ball_logger.endSession();
  }
  if (stopGameLog) {
    std::string lowLevelPath = game_logger.getSessionPath() + "/lowLevel.log";
    stopLoggingLowLevel(lowLevelPath);
    game_logger.endSession();
  }
}

void Robocup::updateBallInformations() {
  std::vector<Eigen::Vector3d> positions;
  // Getting candidates in ball by ROI
  for (const cv::Point3f& ball_pos_in_world : *detectedBalls) {
    try {
      Eigen::Vector3d ballInWorld = cv2Eigen(ball_pos_in_world);
      if (ignoreOutOfFieldBalls && cs->has_camera_field_transform) {
        Eigen::Vector3d ballInField = cs->field_from_camera * cs->worldToCamera * ballInWorld;
        // OPTION: Margin could be added here
        if (!Constants::field.isInArena(cv::Point2f(ballInField.x(), ballInField.y()))) {
          out.warning("Ignoring a ball candidate outside of the field at (%f,%f)", ballInField.x(), ballInField.y());
          continue;
        }
      }
      // Usual code
      positions.push_back(ballInWorld);
      ballSpeedEstimator->update(cs->getTimeStamp(), Eigen::Vector2d(ballInWorld(0), ballInWorld(1)));
    } catch (const std::runtime_error& exc) {
      out.warning("Ignoring a candidate at (%f,%f) because of '%s'", ball_pos_in_world.x, ball_pos_in_world.y,
                  exc.what());
    }
  }
  // Positions are transmitted in the world referential
  ballStackFilter->newFrame(positions);

  // Broadcast information to localisation service
  // Sending data to the loc
  LocalisationService* loc = _scheduler->getServices()->localisation;

  if (ballStackFilter->getCandidates().size() > 0) {
    auto bestCandidate = ballStackFilter->getBest();
    double bsfMaxScore = ballStackFilter->getMaximumScore();
    Point ballSpeed = ballSpeedEstimator->getUsableSpeed();
    loc->setBallWorld(bestCandidate.object, bestCandidate.score / bsfMaxScore, ballSpeed,
                      cs->getTimeStamp());  // This pall pos will be used by BallApproach
  } else {
    loc->setNoBall();
  }

  /// If active: write ballStatus
  if (writeBallStatus) {
    // Some properties are shared for the frame
    double time = getNowTS().getTimeSec();
    for (const Eigen::Vector3d& pos_in_world : positions) {
      Point ball_pos_in_field = loc->worldToField(pos_in_world);
      Point robot_pos = loc->getFieldPos();
      double field_dir = normalizeRad(loc->getFieldOrientation());
      // Entry format:
      // TimeStamp, ballX, ballY, robotX, robotY, fieldDir
      out.log("ballStatusEntry: %lf,%f,%f,%f,%f,%f", time, ball_pos_in_field.x, ball_pos_in_field.y, robot_pos.x,
              robot_pos.y, field_dir);
    }
  }
}

void Robocup::updateRobotInformations() {
  std::vector<Eigen::Vector3d> positions;
  // Getting candidates of current step
  for (const cv::Point3f& robot_pos_in_world : *detectedRobots) {
    try {
      Eigen::Vector3d robotInWorld = cv2Eigen(robot_pos_in_world);
      positions.push_back(robotInWorld);
    } catch (const std::runtime_error& exc) {
      out.warning("Ignoring a candidate at (%f,%f) because of '%s'", robot_pos_in_world.x, robot_pos_in_world.y,
                  exc.what());
    }
  }
  // Positions are transmitted in the world referential
  robotFilter->newFrame(positions);

  // Broadcast information to localisation service
  // Sending data to the loc
  LocalisationService* loc = _scheduler->getServices()->localisation;

  std::vector<Eigen::Vector3d> robot_candidates;
  for (const auto& c : robotFilter->getCandidates()) {
    robot_candidates.push_back(c.object);
  }
  loc->setOpponentsWorld(robot_candidates);
}

std::unique_ptr<Field::POICollection> Robocup::stealFeatures() {
  std::lock_guard<std::mutex> lock(featuresMutex);
  std::unique_ptr<Field::POICollection> tmp = std::move(detectedFeatures);
  detectedFeatures.reset(new Field::POICollection());
  return std::move(tmp);
}

//[Sol]
std::vector<Filters::WhiteLinesData> Robocup::stealWhiteLines() {
  whiteLinesMutex.lock();  // TODO: maybe use visionMutex instead
  whiteLinesCopy = whitelines_data_accumulated;
  whitelines_data_accumulated.clear();
  whiteLinesMutex.unlock();
  // std::cout << "WhiteLinesData.size=" << whiteLinesCopy.size() << std::endl;
  return whiteLinesCopy;
}

void Robocup::stealTags(std::vector<int>& indices, std::vector<Eigen::Vector3d>& positions,
                        std::vector<std::pair<float, float>>& centers,
                        std::vector<std::pair<float, float>>& undistorded_centers, double* timestamp) {
  tagsMutex.lock();
  indices = detectedTagsIndices;
  positions = detectedTagsPositions;
  centers = detectedTagsCenters;
  undistorded_centers = detectedTagsCentersUndistort;
  *timestamp = detectedTimestamp;
  detectedTagsIndices.clear();
  detectedTagsPositions.clear();
  detectedTagsCenters.clear();
  detectedTagsCentersUndistort.clear();
  tagsMutex.unlock();
}

void Robocup::drawDistortedWorldLineOnGround(cv::Mat& img, cv::Point2f p0, cv::Point2f p1, double scale_x,
                                             double scale_y, cv::Scalar color, int thickness, double step) {
  cv::Point2f delta_all = (p1 - p0);
  double delta = step;  // Default step 20cm
  double len = cv::norm(delta_all);
  cv::Point2f p = p0;
  double len_drawn = 0;
  while (len_drawn < len) {
    len_drawn += delta;
    cv::Point2f pa = p;
    p = p + (delta_all / len) * delta;
    cv::Point2f pb = p;
    try {
      cv::Point2f pima = cs->imgXYFromWorldPosition(pa, Vision::Utils::CAMERA_WIDE_FULL);
      pima.x *= scale_x;
      pima.y *= scale_y;
      cv::Point2f pimb = cs->imgXYFromWorldPosition(pb, Vision::Utils::CAMERA_WIDE_FULL);
      pimb.x *= scale_x;
      pimb.y *= scale_y;
      if ((pima.x > 0) && (pima.y > 0) && (pima.x < img.cols) && (pimb.x < img.cols) && (pima.y < img.rows) &&
          (pimb.y < img.rows)) {
        cv::line(img, pima, pimb, color, thickness);
      }
    } catch (const std::runtime_error& exc) {
    }
  }
}

cv::Mat Robocup::getTaggedImg() {
  // cv::Size size = pipeline.get("source").getImg()->size();
  cv::Size size = pipeline.get("sourceRaw").getImg()->size();
  return getTaggedImg(size.width, size.height);
}

/// Black:
/// - ArenaBorders
/// - Goals by ROI (classic)
/// - Number of the ball stack
/// Red:
/// - Direction to opponent goal according to the particle filter
///   - Thick line: direction to the goal
///   - Thin line: direction to the post
/// - Goals By ROI with Tocard detection
/// Blue:
/// - Goals by ROI with standard clipping
/// - The ball detcted
/// - Horizon
/// Cyan:
/// - Position of the ball inside the stack
/// Magenta:
/// - Detected robots
cv::Mat Robocup::getTaggedImg(int width, int height) {
  // TODO: occasionally causes SIGFPE here when looking down on the legs
  if (cs == NULL) throw std::runtime_error("TaggedImg not ready");
  cv::Mat tmp, img, tmp_small;
  globalMutex.lock();

  // tmp = pipeline.get("source").getImg()->clone();
  tmp = pipeline.get("sourceRaw").getImg()->clone();

  cv::Rect2f img_rect(cv::Point2f(), cv::Point2f(width, height));

  cv::resize(tmp, tmp_small, cv::Size(width, height));
  double scale_x = (double)width / (double)tmp.cols;
  double scale_y = (double)height / (double)tmp.rows;

  // cv::cvtColor(tmp_small, img, CV_YCrCb2BGR);
  tmp_small.copyTo(img);  // Spinaker gives RGB as default

  if (pipeline.isFilterPresent("obstacleMap")) {
    // Drawing stereo obstacles
    try {
      Filters::ObstacleMap& obstacleMap = dynamic_cast<Filters::ObstacleMap&>(pipeline.get("obstacleMap"));
      std::vector<Filters::StereoObstacle> detectedObstacles = obstacleMap.getDetectedObstacles();
      for (auto obstacle : detectedObstacles) {
        std::vector<Eigen::Vector2d> corners_in_self = obstacle.hull_points_in_self;
        std::vector<cv::Point> corners_in_img;
        try {
          for (const Eigen::Vector2d& corner_in_self : corners_in_self) {
            Eigen::Vector3d corner_in_world =
                cs->getWorldFromSelf(Eigen::Vector3d(corner_in_self.x(), corner_in_self.y(), 0));
            cv::Point2f p = cs->imgXYFromWorldPosition(corner_in_world, Vision::Utils::CAMERA_WIDE_FULL);
            p.x *= scale_x;
            p.y *= scale_y;
            corners_in_img.push_back(p);
          }
          cv::Mat copy = img.clone();
          cv::Scalar color = cv::Scalar(0, 0, 255);
          cv::fillConvexPoly(copy, corners_in_img, color);
          double alpha = 0.5;  // Transparency of the obstacles
          cv::addWeighted(img, alpha, copy, 1 - alpha, 0, img);
        } catch (std::runtime_error) {
          // do nothing, detected stereo obstacle using model of stereo camera (non-rational by now) is outside of
          // rational model fov used by taggedimg
        }
      }
    } catch (const std::runtime_error& e) {
      out.warning("%s, cannot cast ObstacleMap from pipline for obstacle draw", e.what());
    }
  }

  // Drawing kick areas
  if (taggedKickName != "") {
    const csa_mdp::KickZone& kick_zone = kmc.getKickModel(taggedKickName).getKickZone();
    double alpha = 0.6;  // Transparency of the kick zones
    for (bool is_right_foot : {false, true}) {
      try {
        std::vector<Eigen::Vector2d> corners_in_self = kick_zone.getKickAreaCorners(is_right_foot);
        std::vector<cv::Point> corners_in_img;
        for (const Eigen::Vector2d& corner_in_self : corners_in_self) {
          // std::cout << "corners in self : " << corner_in_self << std::endl;

          Eigen::Vector3d corner_in_world =
              cs->getWorldFromSelf(Eigen::Vector3d(corner_in_self.x(), corner_in_self.y(), 0));

          // std::cout << "corners in World : " << corner_in_world << std::endl;
          cv::Point2f p = cs->imgXYFromWorldPosition(corner_in_world, Vision::Utils::CAMERA_WIDE_FULL);
          p.x *= scale_x;
          p.y *= scale_y;
          corners_in_img.push_back(p);

          // std::cout << "corners in img : " << cs->imgXYFromWorldPosition(corner_in_world) << std::endl;
          // std::cout << "MY XY FROM CAMERA " << cs->imgXYFromSelf(Eigen::Vector3d(corner_in_self.x(),
          // corner_in_self.y(), 0)) << std::endl;
        }
        Eigen::Vector3d wished_pos_in_self = kick_zone.getWishedPos(is_right_foot);
        wished_pos_in_self.z() = 0;
        Eigen::Vector3d wished_pos_in_world = cs->getWorldFromSelf(wished_pos_in_self);
        cv::Point wished_pos_in_img = cs->imgXYFromWorldPosition(wished_pos_in_world, Vision::Utils::CAMERA_WIDE_FULL);
        wished_pos_in_img.x *= scale_x;
        wished_pos_in_img.y *= scale_y;
        // Drawing img
        cv::Mat copy = img.clone();
        cv::Scalar color = is_right_foot ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0);
        cv::fillConvexPoly(copy, corners_in_img, color);
        cv::drawMarker(copy, wished_pos_in_img, 0.2 * color, cv::MARKER_TILTED_CROSS, 15, 2, cv::LINE_AA);
        cv::addWeighted(img, alpha, copy, 1 - alpha, 0, img);
      } catch (const std::runtime_error& exc) {
        // If one of the point is not in image -> ignore kick_zone
        break;
      }
    }
  }

  {
    // Drawing candidates of balls
    auto candidates = ballStackFilter->getCandidates();
    int k = 0;
    for (auto& candidate : candidates) {
      k++;
      Eigen::Vector3d cpos = candidate.object;
      try {
        cv::Point2f pos = cs->imgXYFromWorldPosition(cpos, Vision::Utils::CAMERA_WIDE_FULL);
        pos.x *= scale_x;
        pos.y *= scale_y;

        // I candidate is outside of the image, ignore it
        if (!img_rect.contains(pos)) continue;

        // Draw ball candidate
        cv::circle(img, pos, 6 * candidate.score, cv::Scalar(255, 255, 0), CV_FILLED);

        // Write candidate number
        std::stringstream ss;
        ss << (candidates.size() - k);
        cv::putText(img, ss.str(), pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

        // # Futur position of ball
        // Elapsed time
        double tag_ball_anticipation = 0.2;  // [s] (TODO: set as rhio parameters?)
        double elapsed = tag_ball_anticipation;
        elapsed += diffSec(cs->getTimeStamp(), getNowTS());
        // Compute futur position
        Point ball_usable_speed = ballSpeedEstimator->getUsableSpeed();
        Eigen::Vector3d ball_speed(ball_usable_speed.x, ball_usable_speed.y, 0);

        Eigen::Vector3d next_cpos = cpos + ball_speed * elapsed;
        // cv::Point2f futur_pos = cs->imgXYFromWorldPosition(next_cpos);
        cv::Point2f futur_pos = cs->imgXYFromWorldPosition(next_cpos, Vision::Utils::CAMERA_WIDE_FULL);
        futur_pos.x *= scale_x;
        futur_pos.y *= scale_y;
        cv::line(img, pos, futur_pos, cv::Scalar(255, 255, 0), 2);
      } catch (const std::runtime_error& exc) {
      }
    }
  }

  {
    // Drawing candidates of robots
    auto candidates = robotFilter->getCandidates();
    int k = 0;
    for (auto& candidate : candidates) {
      k++;
      Eigen::Vector3d cpos = candidate.object;
      try {
        cv::Point2f pos = cs->imgXYFromWorldPosition(cv::Point2f(cpos.x(), cpos.y()), Vision::Utils::CAMERA_WIDE_FULL);
        pos.x *= scale_x;
        pos.y *= scale_y;
        // If candidate is outside of the image, ignore it
        if (!img_rect.contains(pos)) continue;

        // Draw robot candidate
        cv::circle(img, pos, 6 * candidate.score, cv::Scalar(255, 0, 255), CV_FILLED);

        // Write candidate number
        std::stringstream ss;
        ss << (candidates.size() - k);
        cv::putText(img, ss.str(), pos, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1);
      } catch (const std::runtime_error& exc) {
        std::cout << "Can not draw robot:" << exc.what() << std::endl;
      }
    }
  }

  // Tagging balls seen on current image with radius
  for (const cv::Point3f& ballPosInWorld : *detectedBalls) {
    try {
      cv::Point center = cs->imgXYFromWorldPosition(cv2Eigen(ballPosInWorld), Vision::Utils::CAMERA_WIDE_FULL);
      double ballRadius = cs->computeBallRadiusFromPixel(center, Vision::Utils::CAMERA_WIDE_FULL);
      if (ballRadius > 0) {
        center.x *= scale_x;
        center.y *= scale_y;
        cv::circle(img, center, (int)(ballRadius * (scale_x + scale_y) / 2.0), cv::Scalar(0, 0, 255), 2);
      }
    } catch (const std::runtime_error& exc) {
      out.warning("%s, cannot find imgXY from world for ballPos: '%s'", DEBUG_INFO, exc.what());
    }
  }

  // Drawing horizon with blue dost
  {
    double angleStep = M_PI / 20;
    std::vector<cv::Point2f> horizonKeypoints;
    Eigen::Vector3d cameraPos = cs->getWorldPosFromCamera(Eigen::Vector3d::Zero());
    Eigen::Vector3d cameraDir = cs->getWorldPosFromCamera(Eigen::Vector3d::UnitZ()) - cameraPos;
    for (double yaw = M_PI; yaw > -M_PI;
         yaw -= angleStep)  // reverse order to make x componnets of points steps from left to right
    {
      Eigen::Vector3d offset(cos(yaw), sin(yaw), 0);
      // skip to next value if object is behind camera plane
      if (cameraDir.dot(offset) <= 0) continue;
      Eigen::Vector3d target = cameraPos + offset;
      try {
        cv::Point2f p = cs->imgXYFromWorldPosition(target, Vision::Utils::CAMERA_WIDE_FULL);
        p.x *= scale_x;
        p.y *= scale_y;
        if (img_rect.contains(p)) {
          horizonKeypoints.push_back(p);
        }
      } catch (const std::runtime_error& exc) {
      }
    }
    if (horizonKeypoints.size() > 0) {
      //Drawing as dots
      for (size_t idx = 0; idx < horizonKeypoints.size() - 1; idx++) {
        cv::circle(img, horizonKeypoints[idx], 3, cv::Scalar(255, 0, 0), -1);
      }

      //Drawing as lines
      double min_x = img.cols;  // Finding from which point to start drawing (input order can be 7 8 9 0 1 2 3 ...)
      int min_x_index = 0;
      for (size_t idx = 0; idx < horizonKeypoints.size(); idx++) {
        if (horizonKeypoints[idx].x < min_x) {
          min_x = horizonKeypoints[idx].x;
          min_x_index = idx;
        }
      }
      for (size_t idx = 0; idx < horizonKeypoints.size() - 1; idx++) {
        cv::Point p0 = horizonKeypoints[(idx + min_x_index) % horizonKeypoints.size()];
        cv::Point p1 = horizonKeypoints[(idx + min_x_index + 1) % horizonKeypoints.size()];
        double length = sqrt(pow(p0.x-p1.x, 2) + pow(p0.y-p1.y, 2));
        if (length < img.cols / 10) //Draw horizon segment only if it's length is reasonably small to protect from erroneous segments due do poor camera calibration etc. It's dumb but it works
          cv::line(img, p0, p1, cv::Scalar(255, 0, 0), 2);
      }
    }
  }

  //[Sol] drawing trunk's middle line(orange) and zero distance line (gray)
  {
    Eigen::Vector3d wa = cs->getWorldFromSelf(Eigen::Vector3d(0, 0, 0));
    Eigen::Vector3d wb = cs->getWorldFromSelf(Eigen::Vector3d(10.0, 0, 0));
    cv::Point2f a = cv::Point2f(wa[0], wa[1]);
    cv::Point2f b = cv::Point2f(wb[0], wb[1]);
    drawDistortedWorldLineOnGround(img, a, b, scale_x, scale_y, cv::Scalar(0, 128, 255), 1, 0.05);

    Eigen::Vector3d wc = cs->getWorldFromSelf(Eigen::Vector3d(0, -0.5, 0));
    Eigen::Vector3d wd = cs->getWorldFromSelf(Eigen::Vector3d(0, 0.5, 0));
    cv::Point2f c = cv::Point2f(wc[0], wc[1]);
    cv::Point2f d = cv::Point2f(wd[0], wd[1]);
    drawDistortedWorldLineOnGround(img, c, d, scale_x, scale_y, cv::Scalar(128, 128, 128), 1, 0.05);
  }

  //[Sol] drawing perspective transform frustum
  {
    Eigen::Vector3d flr;
    Eigen::Vector3d frr;
    Eigen::Vector3d brr;
    Eigen::Vector3d blr;

    cs->getBirdviewRotatedUnionSquareCorners(&flr, &frr, &brr, &blr);

    cv::Point2f self_pos_on_img_fl;
    cv::Point2f self_pos_on_img_fr;
    cv::Point2f self_pos_on_img_bl;
    cv::Point2f self_pos_on_img_br;

    try {
      self_pos_on_img_fl = cs->imgXYFromWorldPosition(flr, Vision::Utils::CAMERA_WIDE_FULL);
      self_pos_on_img_fl.x *= scale_x;
      self_pos_on_img_fl.y *= scale_y;
      circle(img, self_pos_on_img_fl, 5, cv::Vec3b(0, 0, 255), -1);
    } catch (const std::runtime_error& exc) {
    }

    try {
      self_pos_on_img_fr = cs->imgXYFromWorldPosition(frr, Vision::Utils::CAMERA_WIDE_FULL);
      self_pos_on_img_fr.x *= scale_x;
      self_pos_on_img_fr.y *= scale_y;
      circle(img, self_pos_on_img_fr, 5, cv::Vec3b(0, 0, 255), -1);
    } catch (const std::runtime_error& exc) {
    }

    try {
      self_pos_on_img_br = cs->imgXYFromWorldPosition(brr, Vision::Utils::CAMERA_WIDE_FULL);
      self_pos_on_img_br.x *= scale_x;
      self_pos_on_img_br.y *= scale_y;
      circle(img, self_pos_on_img_br, 5, cv::Vec3b(0, 0, 255), -1);
    } catch (const std::runtime_error& exc) {
    }

    try {
      self_pos_on_img_bl = cs->imgXYFromWorldPosition(blr, Vision::Utils::CAMERA_WIDE_FULL);
      self_pos_on_img_bl.x *= scale_x;
      self_pos_on_img_bl.y *= scale_y;
      circle(img, self_pos_on_img_bl, 5, cv::Vec3b(0, 0, 255), -1);
    } catch (const std::runtime_error& exc) {
    }
  }

  // Draw detected corners as yellow dots.
  // Draw corners and lines
  for (size_t i = 0; i < whitelines_data_single_frame.size(); i++) {
    Filters::WhiteLinesData* data = &whitelines_data_single_frame[i];
    if (data->hasCorner) {
      // std::cout << "hasCorner=true" << std::endl;
      cv::Point2f a, b, corner;
      std::vector<std::pair<cv::Point2f, cv::Point2f>> world_lines = data->getLinesInWorldFrame();
      a = world_lines[0].first;
      b = world_lines[1].second;
      corner = world_lines[0].second;

      drawDistortedWorldLineOnGround(img, a, corner, scale_x, scale_y);
      drawDistortedWorldLineOnGround(img, b, corner, scale_x, scale_y);

      try {
        cv::Point2f pa, pb, pcorner;
        pcorner = cs->imgXYFromWorldPosition(corner, Vision::Utils::CAMERA_WIDE_FULL);
        pa.x *= scale_x;
        pa.y *= scale_y;
        pb.x *= scale_x;
        pb.y *= scale_y;
        pcorner.x *= scale_x;
        pcorner.y *= scale_y;
        // cv::line(img, a, corner, cv::Scalar(0, 255, 0), 2);
        // cv::line(img, b, corner, cv::Scalar(0, 255, 0), 2);
        circle(img, pcorner, 4, cv::Vec3b(0, 255, 255), -1);
      } catch (const std::runtime_error& exc) {
      }
    }
    if (data->hasSegment) {
      cv::Point2f a, b;
      std::vector<std::pair<cv::Point2f, cv::Point2f>> world_lines = data->getLinesInWorldFrame();
      a = world_lines[0].first;
      b = world_lines[0].second;

      drawDistortedWorldLineOnGround(img, a, b, scale_x, scale_y);
    }
  }

  /*
  // TODO remove it and do something cleaner lates
  if (cs->has_camera_field_transform)
  {
    // Drawing field_lines
    cv::Mat camera_matrix, distortion_coeffs, rvec, tvec;
    camera_matrix = cs->getCameraModel().getCameraMatrix();
    distortion_coeffs = cs->getCameraModel().getDistortionCoeffs();
    affineToCV(cs->camera_from_field, &rvec, &tvec);

    cv::Scalar line_color(0, 0, 0);
    double line_thickness = 2.0;  // px
    int nb_segments = 10;
    Constants::field.tagLines(camera_matrix, distortion_coeffs, rvec, tvec, &img, line_color, line_thickness,
                              nb_segments);
  }
  */

  //[Sol] drawing FPS
  {
    bool handled = _scheduler->getServices()->decision->handled;
    bool fallen = _scheduler->getServices()->decision->isFallen;
    char s[255];
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 1.1;
    int thickness = 1.5;
    sprintf(s, "Pipeline FPS: %04.1f", pipelineFpsFiltered);
    if (handled) strcat(s, " (handled)");
    if (fallen) strcat(s, " (fallen)");
    cv::putText(img, std::string(s), cv::Point(0, 20), fontFace, fontScale, cv::Vec3b(0, 255, 255), thickness, CV_AA);
    sprintf(s, "Scheduler FPS: %04.1f", moveSchedulerFpsFiltered);
    cv::putText(img, std::string(s), cv::Point(320, 20), fontFace, fontScale, cv::Vec3b(255, 0, 0), thickness, CV_AA);


  }

  globalMutex.unlock();

  return img;
}

cv::Mat Robocup::getRadarImg(int width, int height) {
  if (cs == NULL) throw std::runtime_error("RadarImg not ready");
  cv::Mat img;
  img = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 30, 0));
  // Drawing the robot
  cv::circle(img, cv::Point2i(width / 2, height / 2), 5, cv::Scalar(0, 200, 200), -1);
  globalMutex.lock();

  std::vector<cv::Point2f> freshObservations;
  std::vector<int> delete_me;
  // scale_factor -> conversion [m] -> [px]
  float scale_factor = width / (2 * Constants::field.field_length);
  cv::Scalar ball_color = cv::Scalar(0, 0, 200);
  float discount = 0.05;

  // Drawing static distance marquers each meter (light circles are 0.5 meters)
  for (int i = 1; i < (1 + 2 * Constants::field.field_length); i++) {
    cv::circle(img, cv::Point2i(width / 2, height / 2), (i / 2.0) * scale_factor, cv::Scalar(0, 150, 0), 2 - (i % 2));
  }
  // Drawing vision cone
  rhoban_utils::Angle yaw = cs->getYaw();
  camera_yaw = yaw.getSignedValue();
  rhoban_utils::Angle half_aperture = cs->getCameraModel().getFOVY();
  cv::Point2i p1(width / 2 - height * sin(yaw + half_aperture), height / 2 - height * cos(yaw + half_aperture));
  cv::Point2i p2(width / 2 - height * sin(yaw - half_aperture), height / 2 - height * cos(yaw - half_aperture));

  cv::line(img, cv::Point2i(width / 2, height / 2), p1, cv::Scalar(0, 100, 100), 2);
  cv::line(img, cv::Point2i(width / 2, height / 2), p2, cv::Scalar(0, 100, 100), 2);
  // 0° orientation (front)
  cv::line(img, cv::Point2i(width / 2, height / 2), cv::Point2f(width / 2, 0), cv::Scalar(255, 0, 0), 1);

  for (std::string obsType : observationTypes) {
    std::vector<std::pair<cv::Point2f, float>>& storedObservations = rememberObservations[obsType];
    // Discounting and Killing observations
    delete_me.clear();
    freshObservations.clear();
    for (unsigned int i = 0; i < storedObservations.size(); i++) {
      auto scored_obs = storedObservations[i];
      // Discounting all the old observations
      storedObservations[i].second = storedObservations[i].second - discount;
      if (scored_obs.second < 0) {
        // This observation is considered dead now (checking this only once)
        delete_me.push_back(i);
      }
    }

    for (int i = delete_me.size() - 1; i > -1; i--) {
      // Erasing from the end of the vector to the start, so the smaller indexes
      // don't change.
      storedObservations.erase(storedObservations.begin() + delete_me[i]);
    }

    // Reading the fresh observations (observation type dependent)
    if (obsType == "ball") {
      for (const cv::Point3f& ballPosInWorld : *detectedBalls) {
        freshObservations.push_back(cv::Point2f(ballPosInWorld.x, ballPosInWorld.y));
      }
    } else if (obsType == "tag") {
      // Reading tags
      // Going from and image position to a position on the field, in the origin
      // (of world) frame
      for (unsigned int index = 0; index < detectedTagsCenters.size(); index++) {
        try {
          std::cout << "DetectedTagsCenter at " << detectedTagsCenters[index].first << ", "
                    << detectedTagsCenters[index].second << std::endl;
          // Going from and image position to a position on the field, in the
          // origin (of world) frame
          auto point = cs->worldPosFromImg(detectedTagsCenters[index].first, detectedTagsCenters[index].second);
          freshObservations.push_back(point);
        } catch (const std::runtime_error& exc) {
          // Ignore the candidate
        }
      }
    } else if (obsType == "robot") {
      for (const cv::Point3f& robotPosInWorld : *detectedRobots) {
        freshObservations.push_back(cv::Point2f(robotPosInWorld.x, robotPosInWorld.y));
      }
    } else {
      Field::POIType poiType = Field::string2POIType(obsType);
      for (const cv::Point3f& featurePosInWorld : detectedFeatures->operator[](poiType)) {
        freshObservations.push_back(cv::Point2f(featurePosInWorld.x, featurePosInWorld.y));
      }
    }

    // Adding the observations to the old ones if need be, and updating the intensities
    for (const cv::Point2f& new_obs : freshObservations) {
      storedObservations.push_back(std::pair<cv::Point2f, float>(new_obs, 1.0));
    }

    // Drawing
    for (unsigned int i = 0; i < storedObservations.size(); i++) {
      // Going from meters to pixels, and from the origin frame to the robot one
      // TODO: question, why do we use the max here???
      cv::Point2f obs_in_self = cs->getPosInSelf(storedObservations[i].first);
      cv::Point2f obs_in_img(width / 2 - obs_in_self.y * scale_factor, height / 2 - obs_in_self.x * scale_factor);
      double default_radius = 3;  // [px]
      double marker_size = 8;
      double marker_thickness = 2;
      if (obsType == "robot") {
        cv::circle(img, obs_in_img, default_radius, cv::Scalar(200, 0, 200), -1);
      } else if (obsType == "ball") {
        cv::circle(img, obs_in_img, default_radius, ball_color, -1);
      } else if (obsType == "tag") {
        cv::circle(img, obs_in_img, 5, cv::Scalar(0, 0, 0), -1);
      } else {
        Field::POIType poiType = Field::string2POIType(obsType);
        switch (poiType) {
          case Field::POIType::PostBase:
            cv::circle(img, obs_in_img, default_radius, cv::Scalar(255, 255, 255), -1);
            break;
          case Field::POIType::X:
            cv::drawMarker(img, obs_in_img, cv::Scalar(255, 255, 255), cv::MarkerTypes::MARKER_TILTED_CROSS,
                           marker_size, marker_thickness);
            break;
          case Field::POIType::T:
            cv::drawMarker(img, obs_in_img, cv::Scalar(255, 255, 255), cv::MarkerTypes::MARKER_TRIANGLE_UP, marker_size,
                           marker_thickness);
            break;
          default:
            out.warning("Draw of POI of type '%s' is not implemented", obsType.c_str());
        }
      }
    }
  }

  //[Sol] drawing detected lines/corners in realtime (no persist)
  for (size_t i = 0; i < whitelines_data_single_frame.size(); i++) {
    Filters::WhiteLinesData* data = &whitelines_data_single_frame[i];
    if (data->hasCorner) {
      // std::cout << "hasCorner=true" << std::endl;
      cv::Point2f a, b, corner;
      std::vector<std::pair<cv::Point2f, cv::Point2f>> self_lines = data->getLinesInSelf();
      a = self_lines[0].first;
      b = self_lines[1].second;
      corner = self_lines[0].second;

      cv::Point2f a_in_img(width / 2 - a.y * scale_factor, height / 2 - a.x * scale_factor);
      cv::Point2f b_in_img(width / 2 - b.y * scale_factor, height / 2 - b.x * scale_factor);
      cv::Point2f corner_in_img(width / 2 - corner.y * scale_factor, height / 2 - corner.x * scale_factor);

      cv::line(img, a_in_img, corner_in_img, cv::Scalar(255, 255, 255), 2);
      cv::line(img, b_in_img, corner_in_img, cv::Scalar(255, 255, 255), 2);
      circle(img, corner_in_img, 3, cv::Vec3b(255, 255, 255), -1);
    } else {
      std::pair<cv::Point2f, cv::Point2f> segment = data->getSegmentInSelf();
      cv::Point2f a, b;
      a = segment.first;
      b = segment.second;
      cv::Point2f a_in_img(width / 2 - a.y * scale_factor, height / 2 - a.x * scale_factor);
      cv::Point2f b_in_img(width / 2 - b.y * scale_factor, height / 2 - b.x * scale_factor);
      cv::line(img, a_in_img, b_in_img, cv::Scalar(0, 255, 0), 2);
    }
  }

  //[Sol] draw field lines
  Utils::CameraState* loc_cs;
  // if (isFakeMode()) {
  //[Sol] robocup.cpp and localisationbinding.cpp use different instances of CameraState
  // And in fake mode robocup.cpp doesn't call cs->updateInternalModel which is needed for has_camera_field_transform
  // calculation, so let's use CameraState from localisationbinding here
  //  loc_cs = _scheduler->getServices()->localisation->getLocBinding()->cs;
  //} else {
  loc_cs = cs;
  //}

  // std::cout << "fieldQ=" << _scheduler->getServices()->localisation->fieldQ << std::endl;
  // std::cout << "isFieldQualityGood=" << _scheduler->getServices()->decision->isFieldQualityGood << std::endl;
  // std::cout << "has_camera_field_transform=" << loc_cs->has_camera_field_transform << std::endl;

  if (loc_cs->has_camera_field_transform) {
    const std::vector<Field::Segment>& segments = Constants::field.getWhiteLines();
    for (const Field::Segment& segment : segments) {
      cv::Point3f p0f_cv = segment.first;
      cv::Point3f p1f_cv = segment.second;
      Eigen::Vector3d p0f_eigen = Eigen::Vector3d(p0f_cv.x, p0f_cv.y, p0f_cv.z);
      Eigen::Vector3d p1f_eigen = Eigen::Vector3d(p1f_cv.x, p1f_cv.y, p1f_cv.z);
      Eigen::Vector3d p0w = _scheduler->getServices()->localisation->fieldToWorld(p0f_eigen);
      Eigen::Vector3d p1w = _scheduler->getServices()->localisation->fieldToWorld(p1f_eigen);
      cv::Point2f p0s = cs->getPosInSelf(cv::Point2f(p0w[0], p0w[1]));
      cv::Point2f p1s = cs->getPosInSelf(cv::Point2f(p1w[0], p1w[1]));
      cv::Point2f p0img(width / 2 - p0s.y * scale_factor, height / 2 - p0s.x * scale_factor);
      cv::Point2f p1img(width / 2 - p1s.y * scale_factor, height / 2 - p1s.x * scale_factor);

      cv::line(img, p0img, p1img, cv::Scalar(200, 200, 200), 1);
    }

    /*
    // Drawing field_lines
    cv::Mat camera_matrix, distortion_coeffs, rvec, tvec;
    camera_matrix = cs->getCameraModel(Vision::Utils::CAMERA_WIDE_FULL).getCameraMatrix();
    distortion_coeffs = cs->getCameraModel(Vision::Utils::CAMERA_WIDE_FULL).getDistortionCoeffs();
    affineToCV(cs->camera_from_field, &rvec, &tvec);

    cv::Scalar line_color(0, 0, 0);
    double line_thickness = 2.0;  // px
    int nb_segments = 10;
    Constants::field.tagLines(camera_matrix, distortion_coeffs, rvec, tvec, &img, line_color, line_thickness,
                              nb_segments);
                              */
  }

  globalMutex.unlock();

  return img;
}

cv::Mat Robocup::getApproachImg() {
  cv::Size size = pipeline.get("YBirdview").getImg()->size();
  return getApproachImg(size.width, size.height);
}

cv::Mat Robocup::getApproachImg(int width, int height) {
  // Returning approach debug image with live camera background from birdview

  if (cs == NULL) throw std::runtime_error("ApproachImg not ready");

  if (_scheduler->getMove("approach_potential")->isRunning()) {
    // std::cout << "approach_potential is running" << std::endl;
    cv::Mat birdviewImgGray = pipeline.get("YBirdview").getImg()->clone();

    // Rotate and scale birdview image to make it consistent with approach verbose scale
    rhoban_utils::Angle yaw = cs->getYaw();
    double scale =
        300.0 / (cs->getBirdviewPixelsInOneMeter());  // Magic number taken from ApproachPotential.h->pixInMeter;
    cv::Mat affine = cv::getRotationMatrix2D(cv::Point2f(width / 2, height), yaw.getSignedValue(), scale);
    warpAffine(birdviewImgGray, birdviewImgGray, affine, birdviewImgGray.size());

    cv::Mat birdviewImgColor;
    cv::cvtColor(birdviewImgGray, birdviewImgColor, CV_GRAY2BGR);

    if (!accumulateApproachVerboseOld) {
      // Drawing detected ball taken by it's world coords by small blue circle
      for (cv::Point3f ballWorld : *detectedBalls) {
        cv::Point2f ballSelf = cs->getPosInSelf(cv::Point2f(ballWorld.x, ballWorld.y));
        cv::Point2f ballSelfOnApproachImg(-ballSelf.y * 300.0 + birdviewImgColor.cols / 2,
                                          -ballSelf.x * 300.0 + birdviewImgColor.rows);
        cv::circle(birdviewImgColor, ballSelfOnApproachImg, 10, cv::Scalar(255, 0, 0), 4);
      }

      // Drawing detected ball taken by it's ballStackfilter's world coords by medium cyan circle
      if (ballStackFilter->getCandidates().size() > 0) {
        auto bestCandidate = ballStackFilter->getBest();
        cv::Point2f ballSelf = cs->getPosInSelf(cv::Point2f(bestCandidate.object.x(), bestCandidate.object.y()));
        cv::Point2f ballSelfOnApproachImg(-ballSelf.y * 300.0 + birdviewImgColor.cols / 2,
                                          -ballSelf.x * 300.0 + birdviewImgColor.rows);
        cv::circle(birdviewImgColor, ballSelfOnApproachImg, 15, cv::Scalar(255, 255, 0), 4);
      }

      // Drawing detected ball taken by it's self coords by large yellow circle
      for (cv::Point2f ballSelf : *detectedBallsSelf) {
        cv::Point2f ballSelfOnApproachImg(-ballSelf.y * 300.0 + birdviewImgColor.cols / 2,
                                          -ballSelf.x * 300.0 + birdviewImgColor.rows);
        cv::circle(birdviewImgColor, ballSelfOnApproachImg, 20, cv::Scalar(0, 255, 0), 4);
      }
    }

    std::pair<cv::Point2f, rhoban_utils::Angle> ballOnApproachAccumulatedImg;
    cv::Mat currentApproachImg(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    int typeOfVerbose = 0;
    if (accumulateApproachVerboseOld) typeOfVerbose = 1;  // Draw only robot current position and predicted path
    ballOnApproachAccumulatedImg =
        _scheduler->getMove("approach_potential")->getVerboseImg(currentApproachImg, typeOfVerbose);

    if (accumulateApproachVerbose != accumulateApproachVerboseOld) {
      // accumulateApproachVerbose just changed
      if (accumulateApproachVerbose) {
        // accumulation of ApproachVerboseImg just started
        // birdviewImgColor.copyTo(accumulatedApproachImg);
        accumulatedApproachImg = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::addWeighted(accumulatedApproachImg, 1.0, currentApproachImg, 1.0, 0, accumulatedApproachImg);
        // Remember start ball pos for further accumulation
        ballPosOnApproachAccumulatedImg = ballOnApproachAccumulatedImg.first;
        ballAngleOnApproachAccumulatedImg = ballOnApproachAccumulatedImg.second;
      }
      accumulateApproachVerboseOld = accumulateApproachVerbose;
    } else {
      // accumulateApproachVerbose stays the same
      if (accumulateApproachVerbose) {
        // Accumulate ApproachVerboseImg
        double dx = ballPosOnApproachAccumulatedImg.x - ballOnApproachAccumulatedImg.first.x;
        double dy = ballPosOnApproachAccumulatedImg.y - ballOnApproachAccumulatedImg.first.y;
        double da = (ballAngleOnApproachAccumulatedImg - ballOnApproachAccumulatedImg.second).getSignedValue();
        cv::Mat affine = cv::getRotationMatrix2D(ballOnApproachAccumulatedImg.first, da, 1.0);
        affine.at<double>(0, 2) += dx;
        affine.at<double>(1, 2) += dy;
        warpAffine(currentApproachImg, currentApproachImg, affine, currentApproachImg.size());
        cv::addWeighted(accumulatedApproachImg, 1.0, currentApproachImg, 1.0, 0, accumulatedApproachImg);
      } else {
        // Show recent ApproachVerboseImg without accumulation
        birdviewImgColor.copyTo(accumulatedApproachImg);
        cv::addWeighted(accumulatedApproachImg, 1.0, currentApproachImg, 1.0, 0, accumulatedApproachImg);
      }
    }

    double el = diffSec(approachImgSavedAtTime, getNowTS());
    if ((el >= 0.1) || (approachImgSavedImageNumber == 0)) {  // Save images at 10fps
      if (approachImgSavedImageNumber == 0) {
        approachStartTime = getNowTS();
        // Remember start ball pos for further accumulation
        ballPosOnApproachAccumulatedImg = ballOnApproachAccumulatedImg.first;
        ballAngleOnApproachAccumulatedImg = ballOnApproachAccumulatedImg.second;
      }
      double dx = ballPosOnApproachAccumulatedImg.x - ballOnApproachAccumulatedImg.first.x;
      double dy = ballPosOnApproachAccumulatedImg.y - ballOnApproachAccumulatedImg.first.y;
      double da = (ballAngleOnApproachAccumulatedImg - ballOnApproachAccumulatedImg.second).getSignedValue();
      cv::Mat affine = cv::getRotationMatrix2D(ballOnApproachAccumulatedImg.first, da, 1.0);
      affine.at<double>(0, 2) += dx;
      affine.at<double>(1, 2) += dy;
      warpAffine(accumulatedApproachImg, accumulatedApproachImg, affine, accumulatedApproachImg.size());

      char s[255];
      sprintf(s, "time:%f", diffSec(approachStartTime, getNowTS()));
      int fontFace = cv::FONT_HERSHEY_PLAIN;
      double fontScale = 1.1;
      int thickness = 1;
      cv::putText(accumulatedApproachImg, std::string(s), cv::Point(0, 40), fontFace, fontScale, cv::Scalar::all(255),
                  thickness, CV_AA);

      sprintf(s, "/tmp/approach%03d_%03d.jpg", approachImgSavedApproachNumber, approachImgSavedImageNumber);
      cv::imwrite(s, accumulatedApproachImg);

      approachImgSavedImageNumber++;
      approachImgSavedAtTime = getNowTS();
    }

  } else {
    // Aprroach move is not runing
    if (accumulateApproachVerbose) {
      // Accumulation just finished
      accumulateApproachVerbose = false;
    }
    if (accumulatedApproachImg.cols == 0) {
      // Returning dummy image
      accumulatedApproachImg = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 128, 0));
    }
    if (approachImgSavedImageNumber > 0) {
      approachImgSavedImageNumber = 0;
      approachImgSavedApproachNumber++;
    }
  }
  // std::cout << "accumulateApproachVerbose=" << accumulateApproachVerbose << std::endl;

  return accumulatedApproachImg;
}

std::vector<cv::Point2f> Robocup::keepFrontRobots(std::vector<cv::Point2f>& robots) {
  // TODO, this function should take into account the size of a standard robot and hide any candidate that should be
  // behind another robot
  return robots;
}

void Robocup::run() { launch(); }

void Robocup::closeCamera() {
  if (pipeline.isFilterPresent("source")) {
    std::cerr << "Someone asked to close camera in Robocup, not implemented for PtGrey" << std::endl;
  } else {
    std::cout << "source filter not found (camera might not be closed properly)" << std::endl;
  }
}

TimeStamp Robocup::getNowTS() const {
  if (isFakeMode()) {
    return sourceTS;
  }
  return TimeStamp::now();
}

bool Robocup::isFakeMode() const { return _scheduler->getServices()->model->isFakeMode(); }

void Robocup::ballClear() { ballStackFilter->clear(); }

void Robocup::robotsClear() { robotFilter->clear(); }

void Robocup::ballReset(float x, float y) { ballStackFilter->reset(x, y); }

void Robocup::setLogMode(const std::string& path) {
  _scheduler->getServices()->model->loadReplay(path);

  std::cout << "Loaded replay" << std::endl;
}

void Robocup::setViveLog(const std::string& path) {
  //_scheduler->getServices()->vive->loadLog(path);
}

void Robocup::startLoggingLowLevel(const std::string& path) {
  std::cout << DEBUG_INFO << ": " << path << std::endl;
  _scheduler->getServices()->model->startLogging(path);
}

void Robocup::stopLoggingLowLevel(const std::string& path) {
  out.log("Saving lowlevel log to: %s", path.c_str());
  TimeStamp start_save = TimeStamp::now();
  _scheduler->getServices()->model->stopLogging(path);
  TimeStamp end_save = TimeStamp::now();
  out.log("Lowlevel logs saved in %f seconds", diffSec(start_save, end_save));
}

void Robocup::displaySpecialImagehandlers()
{
  for (SpecialImageHandler& sih : imageHandlers) {
    std::string prefix = "Vision/" + sih.name;
    bool isStreaming = RhIO::Root.frameIsStreaming(prefix);
    // If frame is not displayed either streamed, avoid wasting CPU
    if ((!sih.display) && !isStreaming) continue;
    Benchmark::open(sih.name);
    // Update image and update necessary parts
    int img_width = sih.getWidth();
    int img_height = sih.getHeight();
    sih.lastImg = sih.getter(img_width, img_height);
    if (sih.display) cv::imshow(sih.name, sih.lastImg);
    if (isStreaming) RhIO::Root.framePush(prefix, sih.lastImg);
    Benchmark::close(sih.name.c_str());
  }   
}

int Robocup::getFrames() { return pipeline.frames; }

double Robocup::getLastUpdate() const { return diffMs(lastTS, getNowTS()); }

}  // namespace Vision

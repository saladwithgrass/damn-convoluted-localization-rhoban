#include "CameraState.hpp"
#include <iostream>

#include "Utils/HomogeneousTransform.hpp"
#include "services/DecisionService.h"
#include "services/ModelService.h"
#include "services/LocalisationService.h"
#include "services/RefereeService.h"
//#include "services/ViveService.h"

#include <hl_communication/camera.pb.h>
#include <hl_communication/utils.h>
#include <rhoban_utils/util.h>
#include <rhoban_utils/logging/logger.h>

#include <rhoban_geometry/3d/plane.h>
#include <rhoban_geometry/3d/intersection.h>

#include <cmath>

#include <Eigen/StdVector>
#include <utility>
#include <vector>
#include <sstream>
#include <algorithm>

#include <robocup_referee/constants.h>

#include <string>

using namespace hl_communication;
using namespace hl_monitoring;
using namespace rhoban_geometry;
using namespace rhoban_utils;
using namespace robocup_referee;

static rhoban_utils::Logger logger("CameraState");

namespace Vision {
namespace Utils {
float CameraState::motor_delay(0);

CameraState::CameraState()
    : _moveScheduler(nullptr),
      monotonic_ts(0.0),
      utc_ts(0),
      has_camera_field_transform(false),
      clock_offset(0),
      frame_status(FrameStatus::UNKNOWN_FRAME_STATUS),
      birdviewImageSize(cv::Size(1440 / 2, 1080 / 2)),
      birdviewPixelsInOneMeter(100),
      unionSquareCenterForBirdviewMeters(Eigen::Vector3d(
          1.0, 0.0, 0.0))  // make union square center lying in 1.0m away from robot foot in direction of the camera

{}

CameraState::CameraState(MoveScheduler* moveScheduler) : CameraState() {
  _moveScheduler = moveScheduler;
  _cameraModel = _moveScheduler->getServices()->model->cameraModel;
  _cameraModelWideangleFull = _moveScheduler->getServices()->model->cameraModelWideangleFull;
  _cameraModelWideangleQuarter = _moveScheduler->getServices()->model->cameraModelWideangleQuarter;

  // Making robot center to be at bottom middle point on birdview
  unionSquareCenterOnBirdviewPix =
      cv::Point2f(birdviewImageSize.width / 2 + birdviewPixelsInOneMeter * unionSquareCenterForBirdviewMeters[1],
                  birdviewImageSize.height - birdviewPixelsInOneMeter * (unionSquareCenterForBirdviewMeters[0] - 0.0));
}

CameraState::CameraState(const IntrinsicParameters& camera_parameters, const FrameEntry& frame_entry,
                         const Pose3D& camera_from_self_pose, const Pose3D& camera_from_head_base_pose,
                         const hl_communication::VideoSourceID& source_id_, MoveScheduler* moveScheduler)
    : CameraState(moveScheduler) {
  source_id = source_id_;
  importFromProtobuf(camera_parameters);
  importFromProtobuf(frame_entry);

  Eigen::Affine3d cameraFromSelf = getAffineFromProtobuf(camera_from_self_pose);
  worldToSelf = cameraFromSelf.inverse() * cameraToWorld.inverse();
  selfToWorld = worldToSelf.inverse();
  selfToCamera = cameraFromSelf.inverse();

  cameraFromHeadBase = getAffineFromProtobuf(camera_from_head_base_pose);
  // Removing correction if needed
  if (_moveScheduler == nullptr) {
    throw std::logic_error(DEBUG_INFO + " null movescheduler are not allowed anymore");
  }
  ModelService* modelService = _moveScheduler->getServices()->model;
  if (modelService->applyCorrectionInNonCorrectedReplay) {
    Eigen::Affine3d worldFromHeadBase = cameraToWorld * cameraFromHeadBase;
    cameraToWorld = modelService->applyCalibration(cameraToWorld, worldFromHeadBase, selfToWorld);
    worldToCamera = cameraToWorld.inverse();
    cameraFromHeadBase = worldToCamera * worldFromHeadBase;
  }
  unionSquareCenterOnBirdviewPix =
      cv::Point2f(birdviewImageSize.width / 2 + birdviewPixelsInOneMeter * unionSquareCenterForBirdviewMeters[1],
                  birdviewImageSize.height - birdviewPixelsInOneMeter * (unionSquareCenterForBirdviewMeters[0] - 0.0));
}

cv::Size CameraState::getImgSize() const { return getCameraModel().getImgSize(); }

//[Sol]
cv::Size CameraState::getImgSizeRaw() const { return getCameraModel().getImgSizeRaw(); }

void CameraState::importFromProtobuf(const IntrinsicParameters& camera_parameters) {
  _cameraModel.setCenter(Eigen::Vector2d(camera_parameters.center_x(), camera_parameters.center_y()));
  _cameraModel.setFocal(Eigen::Vector2d(camera_parameters.focal_x(), camera_parameters.focal_y()));
  _cameraModel.setImgWidth(camera_parameters.img_width());
  _cameraModel.setImgHeight(camera_parameters.img_height());
  _cameraModel.setImgWidthRaw(camera_parameters.img_width_raw());
  _cameraModel.setImgHeightRaw(camera_parameters.img_height_raw());
  if (camera_parameters.distortion_size() != 0) {
    Eigen::VectorXd distortion(camera_parameters.distortion_size());
    for (int i = 0; i < camera_parameters.distortion_size(); i++) {
      distortion(i) = camera_parameters.distortion(i);
    }
    _cameraModel.setDistortion(distortion);
  }
}

void CameraState::importFromProtobuf(const FrameEntry& src) {
  monotonic_ts = ((double)src.monotonic_ts()) / std::pow(10, 6);
  utc_ts = src.utc_ts();
  worldToCamera = getAffineFromProtobuf(src.pose());
  cameraToWorld = worldToCamera.inverse();
  selfToWorld = Eigen::Affine3d::Identity();  // Protobuf does not store the position of the robot
  worldToSelf = selfToWorld.inverse();
  if (src.has_status()) {
    frame_status = src.status();
  } else {
    frame_status = FrameStatus::UNKNOWN_FRAME_STATUS;
  }
}

void CameraState::exportToProtobuf(IntrinsicParameters* dst) const {
  dst->set_focal_x(_cameraModel.getFocalX());
  dst->set_focal_y(_cameraModel.getFocalY());
  dst->set_center_x(_cameraModel.getCenterX());
  dst->set_center_y(_cameraModel.getCenterY());
  dst->set_img_width(_cameraModel.getImgWidth());
  dst->set_img_height(_cameraModel.getImgHeight());
  dst->set_img_width_raw(_cameraModel.getImgWidthRaw());
  dst->set_img_height_raw(_cameraModel.getImgHeightRaw());
  Eigen::VectorXd distortion = _cameraModel.getDistortionCoeffsAsEigen();
  dst->clear_distortion();
  for (int i = 0; i < distortion.size(); i++) {
    dst->add_distortion(distortion(i));
  }
}

void CameraState::exportToProtobuf(FrameEntry* dst) const {
  dst->set_monotonic_ts((uint64_t)(monotonic_ts * std::pow(10, 6)));
  dst->set_utc_ts(utc_ts);
  setProtobufFromAffine(worldToCamera, dst->mutable_pose());
  dst->set_status(frame_status);
}

void CameraState::importHeader(const hl_communication::VideoMetaInformation& src) {
  if (src.has_camera_parameters()) {
    importFromProtobuf(src.camera_parameters());
  }
  if (src.has_source_id()) {
    source_id.CopyFrom(src.source_id());
  }
}

void CameraState::exportHeader(hl_communication::VideoMetaInformation* dst) const {
  exportToProtobuf(dst->mutable_camera_parameters());
  dst->mutable_source_id()->CopyFrom(source_id);
}

const rhoban::CameraModel& CameraState::getCameraModel(VirtualCameraTypeEnum cameraType) const {
  switch (cameraType) {
    case CAMERA_DEFAULT_NARROW_SMALL:
      return _cameraModel;
      break;
    case CAMERA_WIDE_FULL:
      return _cameraModelWideangleFull;
      break;
    case CAMERA_WIDE_QUARTER:
      return _cameraModelWideangleQuarter;
      break;
    default:
      throw std::runtime_error(DEBUG_INFO + "Unknown cameraType");  // Weird because we have enum
  }
  // return _cameraModel; //Even weirder case
}

double CameraState::getSchedulerTS() const { return getSchedulerTS(getTimeStamp()); }

double CameraState::getSchedulerTS(const rhoban_utils::TimeStamp& ts) const {
  return ((double)ts.getTimeUS(false) / 1000.0 + motor_delay) / 1000.0;
}

void CameraState::updateInternalModel(const rhoban_utils::TimeStamp& ts) {
  monotonic_ts = ts.getTimeUS(false);
  utc_ts = ts.getTimeUS(true);

  if (_moveScheduler != nullptr) {
    double scheduler_ts = getSchedulerTS(ts);

    //printf("Picking cameraToWorld from %f, camera state ts=%f\r\n", scheduler_ts, ts.getTimeMS(false) / 1000.0);

    ModelService* modelService = _moveScheduler->getServices()->model;
    // ViveService* vive = _moveScheduler->getServices()->vive;
    DecisionService* decision = _moveScheduler->getServices()->decision;
    RefereeService* referee = _moveScheduler->getServices()->referee;

    source_id.Clear();
    RobotCameraIdentifier* identifier = source_id.mutable_robot_source();
    identifier->mutable_robot_id()->set_team_id(referee->teamId);
    identifier->mutable_robot_id()->set_robot_id(referee->id);
    identifier->set_camera_name("main");

    selfToWorld = modelService->selfToWorld(scheduler_ts);
    worldToCamera = modelService->cameraToWorld(scheduler_ts).inverse();
    _cameraModel = modelService->cameraModel;
    worldToSelf = selfToWorld.inverse();
    cameraToWorld = worldToCamera.inverse();
    // std::cout << "cameraToWorld" << cameraToWorld.matrix() << std::endl;
    cameraFromHeadBase = worldToCamera * modelService->headBaseToWorld(scheduler_ts).inverse();
    frame_status = decision->camera_status;
    // Update camera/field transform based on (by order of priority)
    // 1. Vive
    // 2. LocalisationService if field quality is good
    // 3. If nothing is available set info to false
    /*
    vive_balls_in_field.clear();
        vive_trackers_in_field.clear();
        if (vive->isActive())
        {
          try
          {
            camera_from_field = vive->getFieldToCamera(utc_ts, true);
            std::cout << "Vive based update: cameraPosInField: "
                      << (camera_from_field.inverse() * Eigen::Vector3d::Zero()).transpose() << std::endl;
            has_camera_field_transform = true;
            for (const Eigen::Vector3d& tagged_pos : vive->getTaggedPositions(utc_ts, true))
            {
              Eigen::Vector3d ball_pos = tagged_pos;
              ball_pos.z() = Constants::field.ball_radius;
              vive_balls_in_field.push_back(ball_pos);
            }
            for (const Eigen::Vector3d& pos : vive->getOthersTrackersPos(utc_ts, true))
            {
              vive_trackers_in_field.push_back(pos);
            }
          }
          catch (const std::out_of_range& exc)
          {
            has_camera_field_transform = false;
            camera_from_field = Eigen::Affine3d::Identity();
            logger.error("Failed to import transform from Vive: %s", exc.what());
          }
          catch (const std::runtime_error& exc)
          {
            has_camera_field_transform = false;
            camera_from_field = Eigen::Affine3d::Identity();
            logger.error("Failed to import transform from Vive: %s", exc.what());
          }
        }*/

    // std::cout << "cs->isFieldQualityGood=" << decision->isFieldQualityGood << std::endl;
    // std::cout << "cs->has_camera_field_transform=" << has_camera_field_transform << std::endl;
    // std::cout << "cs->cs=" << this << std::endl;

    if (decision->isFieldQualityGood) {
      LocalisationService* loc = _moveScheduler->getServices()->localisation;
      camera_from_field = worldToCamera * loc->world_from_field;
      has_camera_field_transform = true;
    } else {
      has_camera_field_transform = false;
      camera_from_field = Eigen::Affine3d::Identity();
    }
    field_from_camera = camera_from_field.inverse();
    // std::cout << "cs->has_camera_field_transform=" << has_camera_field_transform << std::endl;
  } else {
    logger.warning("Not updating internal model (no moveScheduler available)");
  }
}

Angle CameraState::getTrunkYawInWorld() {
  Eigen::Vector3d dirInWorld = selfToWorld.linear() * Eigen::Vector3d::UnitX();

  return Angle(rad2deg(atan2(dirInWorld(1), dirInWorld(0))));
}

cv::Point2f CameraState::robotPosFromImg(double imgX, double imgY, VirtualCameraTypeEnum cameraType) const {
  cv::Point2f posInWorldCV = worldPosFromImg(imgX, imgY, cameraType);

  Eigen::Vector3d posInWorld(posInWorldCV.x, posInWorldCV.y, 0);
  Eigen::Vector3d posInSelf = worldToSelf * posInWorld;

  return cv::Point2f(posInSelf(0), posInSelf(1));
}

cv::Point2f CameraState::worldPosFromImg(double imgX, double imgY, VirtualCameraTypeEnum cameraType) const {
  Eigen::Vector3d posInWorld = posInWorldFromPixel(cv::Point2f(imgX, imgY), 0.0, cameraType);

  return cv::Point2f(posInWorld(0), posInWorld(1));
}

Eigen::Vector2d CameraState::getVecInSelf(const Eigen::Vector2d& vec_in_world) const {
  Eigen::Vector3d src_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d dst_in_world = Eigen::Vector3d::Zero();
  dst_in_world.segment(0, 2) = vec_in_world;

  Eigen::Vector3d src_in_self, dst_in_self;
  src_in_self = worldToSelf * src_in_world;
  dst_in_self = worldToSelf * dst_in_world;

  return (dst_in_self - src_in_self).segment(0, 2);
}

cv::Point2f CameraState::getPosInSelf(const cv::Point2f& pos_in_origin) const {
  Eigen::Vector3d pos_in_self = worldToSelf * Eigen::Vector3d(pos_in_origin.x, pos_in_origin.y, 0);
  return cv::Point2f(pos_in_self(0), pos_in_self(1));
}

rhoban_geometry::PanTilt CameraState::robotPanTiltFromImg(double imgX, double imgY) const {
  Eigen::Vector3d viewVectorInCamera = cv2Eigen(_cameraModel.getViewVectorFromImg(cv::Point2f(imgX, imgY)));
  Eigen::Vector3d viewVectorInSelf = worldToSelf.linear() * cameraToWorld.linear() * viewVectorInCamera;

  return rhoban_geometry::PanTilt(viewVectorInSelf);
}

Eigen::Vector3d CameraState::getWorldPosFromCamera(const Eigen::Vector3d& pos_camera) const {
  return cameraToWorld * pos_camera;
}

Eigen::Vector3d CameraState::getSelfFromWorld(const Eigen::Vector3d& pos_world) const {
  return worldToSelf * pos_world;
}

cv::Point2f CameraState::imgXYFromSelf(const Eigen::Vector3d& pos_self, VirtualCameraTypeEnum cameraType) const {
  Eigen::Vector3d posInCamera = selfToCamera * pos_self;
  return getCameraModel(cameraType).getImgFromObject(eigen2CV(posInCamera));
}

Eigen::Vector3d CameraState::getWorldFromSelf(const Eigen::Vector3d& pos_self) const { return selfToWorld * pos_self; }

Angle CameraState::getPitch() {
  PanTilt panTilt(worldToSelf.linear() * cameraToWorld.linear() * Eigen::Vector3d::UnitZ());
  return panTilt.tilt;
}

Angle CameraState::getYaw() {
  PanTilt panTilt(worldToSelf.linear() * cameraToWorld.linear() * Eigen::Vector3d::UnitZ());
  return panTilt.pan;
}

double CameraState::getHeight() {
  // Getting height at camera origin
  double height = (cameraToWorld * Eigen::Vector3d::Zero())(2);
  if (height < 0) {
    height = 0;
  }
  return height;
}

cv::Point2f CameraState::imgXYFromWorldPosition(const cv::Point2f& p, VirtualCameraTypeEnum cameraType) const {
  return imgXYFromWorldPosition(Eigen::Vector3d(p.x, p.y, 0), cameraType);
}

cv::Point2f CameraState::imgXYFromWorldPosition(const Eigen::Vector3d& posInWorld,
                                                VirtualCameraTypeEnum cameraType) const {
  Eigen::Vector3d posInCamera = worldToCamera * posInWorld;
  return getCameraModel(cameraType).getImgFromObject(eigen2CV(posInCamera));
}

cv::Point2f CameraState::imgFromFieldPosition(const Eigen::Vector3d& pos_in_field) const {
  if (!has_camera_field_transform) {
    throw std::runtime_error(DEBUG_INFO + "no camera_field_transform available");
  }
  Eigen::Vector3d point_in_camera = camera_from_field * pos_in_field;
  return getCameraModel().getImgFromObject(eigen2CV(point_in_camera));
}

PanTilt CameraState::panTiltFromXY(const cv::Point2f& pos, double height) {
  return PanTilt(Eigen::Vector3d(pos.x, pos.y, -height));
}

double CameraState::computeBallRadiusFromPixel(const cv::Point2f& ballPosImg, VirtualCameraTypeEnum cameraType) const {
  
  //if(!getCameraModel(cameraType).containsPixel(ballPosImg)) {
  //  throw std::logic_error("containsPixel==false!");
  //}
  //std::cout << "(" << ballPosImg.x << "," << ballPosImg.y << ")" << std::endl;
  // TODO: occasionally causes SIGFPE here in case of bad camera calibration etc
  Ray viewRay = getRayInWorldFromPixel(ballPosImg, cameraType);
  if (viewRay.dir.z() >= 0) {
    return -1;
  }

  Plane ballPlane(Eigen::Vector3d::UnitZ(), Constants::field.ball_radius * 1.65);

  if (!isIntersectionPoint(viewRay, ballPlane)) {
    return -1;
  }

  Eigen::Vector3d ballCenter = getIntersection(viewRay, ballPlane);

  // Getting a perpendicular direction. We know that viewRay.dir.z<0, thus the
  // vectors will be different
  Eigen::Vector3d groundDir = viewRay.dir;
  groundDir(2) = 0;
  Eigen::Vector3d altDir = viewRay.dir.cross(groundDir).normalized();

  // Getting pixel for ballSide
  double side_sum = 0;
  int nb_points = 0;
  // Testing two directions because one can be out of the image
  for (int side : {-1, 1}) {
    // This is not an exact method, but the approximation should be good enough
    Eigen::Vector3d ballSide = ballCenter + side * altDir * Constants::field.ball_radius;
    if (ballSide.z() <= 0) {
      ballSide.z() = 0.01;
    }

    // pay attention that we are using float pixel coord below, not int (Point2i).
    // Using int pixel coords here will throw SIGFPE when converting float large enough not to fit in int (i.e. -1.00908176e+14 as in last crash debug ) in saturate_cast
    cv::Point2f ballSideImgFloat;
    try {
      ballSideImgFloat = imgXYFromWorldPosition(ballSide, cameraType);
    } catch (const std::runtime_error& exc) {
      return -1;
    }
    // Then we are checking if resulting world->image position lies in the image  
    if(!getCameraModel(cameraType).containsPixel(ballSideImgFloat)) 
      return -1; 
    // Now we can safely cast float pixel coords to int
    cv::Point2i ballSideImg = ballSideImgFloat; 
    side_sum += (cv2Eigen(ballPosImg) - cv2Eigen(ballSideImg)).norm();
    nb_points++;
  }
  if (nb_points == 0) return -1;
  return side_sum / nb_points;
}

Eigen::Vector3d CameraState::ballInWorldFromPixel(const cv::Point2f& pos, VirtualCameraTypeEnum cameraType) const {
  return posInWorldFromPixel(pos, Constants::field.ball_radius, cameraType);
  // return posInWorldFromPixel(pos, 0, cameraType); //[Sol] it's strange, but debug based on ApproachImg shows that we
  // shoud use zero height here. Live tests pending
}

rhoban_geometry::Ray CameraState::getRayInWorldFromPixel(
        const cv::Point2f& img_pos,
        VirtualCameraTypeEnum cameraType
) const {
  Eigen::Vector3d viewVectorInCamera = cv2Eigen(getCameraModel(cameraType).getViewVectorFromImg(img_pos));
  Eigen::Vector3d viewVectorInWorld = cameraToWorld.linear() * viewVectorInCamera;

  Eigen::Vector3d cameraPosInWorld = cameraToWorld * Eigen::Vector3d::Zero();

  return Ray(cameraPosInWorld, viewVectorInWorld);
}

Eigen::Vector3d CameraState::posInWorldFromPixel(const cv::Point2f& pos, double ground_height,
                                                 VirtualCameraTypeEnum cameraType) const {
  Ray viewRay = getRayInWorldFromPixel(pos, cameraType);
  Plane groundPlane(Eigen::Vector3d(0, 0, 1), ground_height);

  if (!isIntersectionPoint(viewRay, groundPlane)) {
    std::ostringstream oss;
    oss << DEBUG_INFO << " Point x=" << pos.x << ", y=" << pos.y
        << " does not intersect ground, cameraType=" << cameraType << std::endl;
    throw std::runtime_error(oss.str());
  }

  return getIntersection(viewRay, groundPlane);
}

::rhoban_utils::TimeStamp CameraState::getTimeStamp() const {
  return ::rhoban_utils::TimeStamp::fromMS(monotonic_ts / 1000);
}

uint64_t CameraState::getTimeStampUs() const { return (uint64_t)(monotonic_ts * std::pow(10, 6)); }

double CameraState::getTimeStampDouble() const { return monotonic_ts / 1000000.0; }

void CameraState::setClockOffset(int64_t new_offset) { clock_offset = new_offset; }

//[Sol] retrofited from 2018
double CameraState::getPixelYtAtHorizon(double pixelX, double imgWidth, double imgHeight) {
  /*
    //TODO: should we undistort this?
    double x = (pixelX - imgWidth / 2) * 2 / imgWidth;
    double y = _cameraModel->cameraScreenHorizon(_params, x);
    return (imgHeight / 2) * y + imgHeight / 2;
  */
}

//[Sol] retrofitted from 2018
/*cv::Point CameraState::imgXYFromRobotPosition(const cv::Point2f &p,
                                              double imgWidth, double imgHeight,
                                              bool self) {


  Eigen::Vector2d pixel(0.0, 0.0);
  cv::Point2f distorted(-1.0, -1.0); //In case of false from cameraWorldToPixel
  Eigen::Vector3d pos(p.x, p.y, 0);
  // pos is expressed in the robot's frame, we need it in the world's frame
  if (self) {
    //pos = _Model->selfInFrame("origin", pos);
    // Use Model and camera parameters to determine position
    rhoban::HumanoidModel* model = &(_moveScheduler->getServices()->model->model);
    pos = model->selfToWorld() * pos; //2019 version
  }
  bool success = _cameraModel->cameraWorldToPixel(_params, pos, pixel);
  if (!success) {
    return cv::Point(-1,-1);
  } else {
    //pixel is in [-1, 1] => remap in [0, imgHeight]
    distortPoint((pixel(0)+1.0)/2.0*imgWidth, (pixel(1)+1.0)/2.0* imgHeight, imgWidth, imgHeight, distorted);
  }

  // // Pixel is in [-1, 1]
  // return cv::Point(distorted.x * imgWidth / 2 + imgWidth / 2,
  //                  distorted.y * imgHeight / 2 + imgHeight / 2);

  return cv::Point(distorted.x ,
                   distorted.y );

}

//[Sol] retrofitted from 2018
cv::Point2f CameraState::imgXYFromRobotPosition2f(const cv::Point2f &p,
                                              double imgWidth, double imgHeight,
                                              bool self) {


  Eigen::Vector2d pixel(0.0, 0.0);
  cv::Point2f distorted(-1.0, -1.0); //In case of false from cameraWorldToPixel
  Eigen::Vector3d pos(p.x, p.y, 0);
  // pos is expressed in the robot's frame, we need it in the world's frame
  if (self) {
    pos = _cameraModel->selfInFrame("origin", pos);
  }
  bool success = _cameraModel->cameraWorldToPixel(_params, pos, pixel);
  if (!success) {
    return cv::Point(-1,-1);
  } else {
    //pixel is in [-1, 1] => remap in [0, imgHeight]
    distortPoint((pixel(0)+1.0)/2.0*imgWidth, (pixel(1)+1.0)/2.0* imgHeight, imgWidth, imgHeight, distorted);
  }

  // // Pixel is in [-1, 1]
  // return cv::Point(distorted.x * imgWidth / 2 + imgWidth / 2,
  //                  distorted.y * imgHeight / 2 + imgHeight / 2);

  return cv::Point2f(distorted.x ,
                   distorted.y );

}
*/

//[Sol]
int CameraState::lineInfoFromPixel(
        const cv::Point2f& pos, 
        float* dx, float* dy, 
        int* px0, int* py0, 
        int* px1, int* py1,
        int* px2, int* py2, 
        int* px3, int* py3, 
        double angularPitchError) {
  // Ray cameraCentralRay = getRayInWorldFromPixel(cv::Point2f(getImgSize().width/2, getImgSize().height/2));

  Ray cameraCentralRay = getRayInWorldFromPixel(cv::Point2f(getImgSize().width / 2, 0));
  // Ray cameraCentralRay = getRayInWorldFromPixel(pos);
  // Ray consist of source and dir, both Eigen3D

  Eigen::Vector3d forwardDir = Eigen::Vector3d(cameraCentralRay.dir(0), cameraCentralRay.dir(1), 0.0).normalized();
  Eigen::Vector3d leftDir = Eigen::Vector3d(cameraCentralRay.dir(1), -cameraCentralRay.dir(0), 0.0).normalized();
  // Eigen::Vector3d leftDir = Eigen::Vector3d(0.0, 0.0, 0.0);
  // Eigen::Vector3d leftDir = viewRay.dir.cross(groundDir).normalized();

  Ray viewRay = getRayInWorldFromPixel(pos);
  if (viewRay.dir.z() >= 0)  // Point above horison (z componnet of direction >0)
  {
    return -1;
  }
  Plane groundPlane(Eigen::Vector3d(0, 0, 1), 0.0);

  if (!isIntersectionPoint(viewRay, groundPlane)) {
    return -1;
  }

  Eigen::Vector3d posInWorld = getIntersection(viewRay, groundPlane);
  cv::Point2f pof = cv::Point2f(posInWorld(0), posInWorld(1));

  cv::Point2f shiftxr = cv::Point2f(leftDir(0) * 0.05 / 2, leftDir(1) * 0.05 / 2);
  cv::Point2f shiftyr = cv::Point2f(forwardDir(0) * 0.05 / 2, forwardDir(1) * 0.05 / 2);

  /*
  //Calculating vector to shift pixel to half of line width in "x" (from left to right) direction in image taking into
  account camera Yaw cv::Point2f shiftx = cv::Point2f(0.0, 0.05/2.0); //Shift to the left cv::Point2f shiftxr =
  cv::Point2f(shiftx.x * cos(getYaw()) - shiftx.y * sin(getYaw()), shiftx.x * sin(getYaw()) + shiftx.y * cos(getYaw())
  );

  //Calculating vector to shift pixel to half of line width in "y" (from bottom to top) direction in image taking into
  account camera Yaw
  //cv::Point2f shifty = cv::Point2f(0.05/2.0, 0.00); //Shift forward
  cv::Point2f shifty = cv::Point2f(0.0, 0.00); //Shift forward
  cv::Point2f shiftyr = cv::Point2f(shifty.x * cos(getYaw()) - shifty.y * sin(getYaw()),
                                   shifty.x * sin(getYaw()) + shifty.y * cos(getYaw()) );

  //cv::Point2f pof = robotPosFromImg(pos.x, pos.y); //Get point on field from pixel coords
  cv::Point2f pof = worldPosFromImg(pos.x, pos.y); //Get point on field from pixel coords, 2019 version
  */
  cv::Point2f pim0sx, pim1sx, pim0sy, pim1sy;

  try {
    pim0sx = imgXYFromWorldPosition(pof - shiftxr);  // Get pixel coords of shifted point on field
    pim1sx = imgXYFromWorldPosition(pof + shiftxr);  // Get pixel coords of shifted point on field
    pim0sy = imgXYFromWorldPosition(pof - shiftyr);  // Get pixel coords of shifted point on field
    pim1sy = imgXYFromWorldPosition(pof + shiftyr);  // Get pixel coords of shifted point on field
  } catch (const std::runtime_error& exc) {
    return -1;
  }

  /*
  //cv::Point2f pim0sx = imgXYFromRobotPosition2f(pof - shiftxr, width, height); //Get pixel coords of shifted point on
  field
  //cv::Point2f pim1sx = imgXYFromRobotPosition2f(pof + shiftxr, width, height); //Get pixel coords of shifted point on
  field
  //cv::Point2f pim0sy = imgXYFromRobotPosition2f(pof - shiftyr, width, height); //Get pixel coords of shifted point on
  field
  //cv::Point2f pim1sy = imgXYFromRobotPosition2f(pof + shiftyr, width, height); //Get pixel coords of shifted point on
  field
  */

  *px0 = pim0sx.x;
  *py0 = pim0sx.y;
  *px1 = pim1sx.x;
  *py1 = pim1sx.y;
  *px2 = pim0sy.x;
  *py2 = pim0sy.y;
  *px3 = pim1sy.x;
  *py3 = pim1sy.y;

  // *dx = fabs(pim0sx.x - pim1sx.x);
  // *dy = fabs(pim0sy.y - pim1sy.y);
  *dx = sqrt(pow(pim0sx.x - pim1sx.x, 2) + pow(pim0sx.y - pim1sx.y, 2));
  *dy = sqrt(pow(pim0sy.x - pim1sy.x, 2) + pow(pim0sy.y - pim1sy.y, 2));

  return 0;  // no errors
}

//[Sol]
cv::Point2f CameraState::robotPosFromBirdviewImg(double imgX, double imgY) {
  // Getting coords in m in birdview frame rotated by camera yaw
  Eigen::Vector3d rotatedRobotPos;
  cv::Point2f unionSquareCenterOnBirdviewPix = getUnionSquareCenterOnBirdviewPix();
  rotatedRobotPos[0] =
      -(imgY - unionSquareCenterOnBirdviewPix.y) / birdviewPixelsInOneMeter + unionSquareCenterForBirdviewMeters[0];
  rotatedRobotPos[1] =
      -(imgX - unionSquareCenterOnBirdviewPix.x) / birdviewPixelsInOneMeter + unionSquareCenterForBirdviewMeters[1];
  rotatedRobotPos[2] = unionSquareCenterForBirdviewMeters[2];  // Shoud be zero, but for code clarity

  // Unrotate
  double cameraYaw = getYaw().getSignedValue();
  Eigen::Affine3d a = Eigen::Affine3d::Identity();
  a.linear() = Eigen::AngleAxisd(cameraYaw * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Eigen::Vector3d unrotatedRobotPos = a * rotatedRobotPos;
  return cv::Point2f(unrotatedRobotPos(0), unrotatedRobotPos(1));
}

cv::Point2f CameraState::worldPosFromBirdviewImg(double imgX, double imgY) {
  cv::Point2f posInSelfCV = robotPosFromBirdviewImg(imgX, imgY);
  Eigen::Vector3d posInSelf(posInSelfCV.x, posInSelfCV.y, 0);
  Eigen::Vector3d posInWorld = selfToWorld * posInSelf;

  return cv::Point2f(posInWorld(0), posInWorld(1));
}

void CameraState::getBirdviewRotatedUnionSquareCorners(Eigen::Vector3d* flr, Eigen::Vector3d* frr, Eigen::Vector3d* brr,
                                                       Eigen::Vector3d* blr) {
  Eigen::Vector3d fl_self_unrotated = unionSquareCenterForBirdviewMeters + Eigen::Vector3d(0.5, 0.5, 0);
  Eigen::Vector3d fr_self_unrotated = unionSquareCenterForBirdviewMeters + Eigen::Vector3d(0.5, -0.5, 0);
  Eigen::Vector3d br_self_unrotated = unionSquareCenterForBirdviewMeters + Eigen::Vector3d(-0.5, -0.5, 0);
  Eigen::Vector3d bl_self_unrotated = unionSquareCenterForBirdviewMeters + Eigen::Vector3d(-0.5, 0.5, 0);

  double cameraYaw = getYaw().getSignedValue();
  Eigen::Affine3d a = Eigen::Affine3d::Identity();
  a.linear() = Eigen::AngleAxisd(cameraYaw * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  Eigen::Vector3d fl_self_rotated = a * fl_self_unrotated;
  Eigen::Vector3d fr_self_rotated = a * fr_self_unrotated;
  Eigen::Vector3d br_self_rotated = a * br_self_unrotated;
  Eigen::Vector3d bl_self_rotated = a * bl_self_unrotated;

  Eigen::Vector3d fl = getWorldFromSelf(fl_self_rotated);
  Eigen::Vector3d fr = getWorldFromSelf(fr_self_rotated);
  Eigen::Vector3d br = getWorldFromSelf(br_self_rotated);
  Eigen::Vector3d bl = getWorldFromSelf(bl_self_rotated);

  *flr = fl;
  *frr = fr;
  *brr = br;
  *blr = bl;
}

double CameraState::getBirdviewPixelsInOneMeter() const { return birdviewPixelsInOneMeter; }

cv::Size CameraState::getBirdviewImageSize() const { return birdviewImageSize; }

cv::Point2f CameraState::getUnionSquareCenterOnBirdviewPix() const { return unionSquareCenterOnBirdviewPix; }

}  // namespace Utils
}  // namespace Vision

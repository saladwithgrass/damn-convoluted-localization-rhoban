#pragma once

#include "scheduler/MoveScheduler.h"

#include <hl_communication/camera.pb.h>
#include <opencv2/core/core.hpp>
#include <rhoban_geometry/3d/ray.h>
#include <rhoban_geometry/3d/pan_tilt.h>
#include <rhoban_utils/angle.h>
#include <rhoban_utils/timing/time_stamp.h>
#include <robot_model/camera_model.h>

#include <utility>
#include <string>
#include <stdexcept>

namespace Vision
{
namespace Utils
{
/// Relevant basis:
/// - World: fixed reference in which the camera is evolving
/// - Self: A basis centered on the robot
///   - Center: Projection of trunk of the robot on the ground
///   - X-axis: in front of the robot
///   - Y-axis: left of the robot
///   - Z-axis: same as world axis
/// - Camera:
///   - Center: At camera optical center
///   - X-axis: aligned with x-axis of the image
///   - Y-axis: aligned with y-axis of the image
///   - Z-axis: direction toward which the camera is pointing

enum VirtualCameraTypeEnum {
  CAMERA_DEFAULT_NARROW_SMALL, //0
  CAMERA_WIDE_FULL, //1
  CAMERA_WIDE_QUARTER //2
} ;

class CameraState
{
public:
  CameraState();
  CameraState(MoveScheduler* moveScheduler);
  CameraState(const hl_communication::IntrinsicParameters& camera_parameters,
              const hl_communication::FrameEntry& frame_entry, const hl_communication::Pose3D& camera_from_self_pose,
              const hl_communication::Pose3D& camera_from_head_base_pose,
              const hl_communication::VideoSourceID& source_id, MoveScheduler* moveScheduler);

  cv::Size getImgSize() const;
  cv::Size getImgSizeRaw() const; //[Sol]

  void importFromProtobuf(const hl_communication::IntrinsicParameters& camera_parameters);
  void importFromProtobuf(const hl_communication::FrameEntry& src);
  void exportToProtobuf(hl_communication::IntrinsicParameters* dst) const;
  void exportToProtobuf(hl_communication::FrameEntry* dst) const;
  /**
   * Import source_id and camera_parameters from the VideoMetaInformation
   */
  void importHeader(const hl_communication::VideoMetaInformation& src);
  /**
   * Export source_id and camera_parameters to the VideoMetaInformation
   */
  void exportHeader(hl_communication::VideoMetaInformation* dst) const;

  const rhoban::CameraModel& getCameraModel(VirtualCameraTypeEnum cameraType = CAMERA_DEFAULT_NARROW_SMALL) const;

  /// Asks the model to update itself to the state the robot had at given timestamps
  /// Both monotonic and utc_ts should be provided
  void updateInternalModel(const rhoban_utils::TimeStamp& ts);

  /// Return the [x,y] position of the ground point seen at (imgX, imgY)
  /// in self referential [m]
  /// throws a runtime_error if the point requested is above horizon
  cv::Point2f robotPosFromImg(double imgX, double imgY, VirtualCameraTypeEnum cameraType = CAMERA_DEFAULT_NARROW_SMALL) const;

  /// Return the [x,y] position of the ground point seen at (imgX, imgY)
  /// in world referential [m]
  /// throws a runtime_error if the point requested is above horizon
  cv::Point2f worldPosFromImg(double imgX, double imgY, VirtualCameraTypeEnum cameraType = CAMERA_DEFAULT_NARROW_SMALL) const;

  /// Converting vector from world referential to self referential
  Eigen::Vector2d getVecInSelf(const Eigen::Vector2d& vec_in_world) const;

  /// Return the position in the robot basis from a position in origin basis
  cv::Point2f getPosInSelf(const cv::Point2f& pos_in_origin) const;

  /// Return the [pan, tilt] pair of the ground point seen at imgX, imgY
  rhoban_geometry::PanTilt robotPanTiltFromImg(double imgX, double imgY) const;

  /// Convert the position 'pos_camera' (in camera referential) to the 'world' basis
  Eigen::Vector3d getWorldPosFromCamera(const Eigen::Vector3d& pos_camera) const;

  /// Convert the position 'pos_world' (in world referential) to the 'self' basis
  Eigen::Vector3d getSelfFromWorld(const Eigen::Vector3d& pos_world) const;

  /// Convert the position 'pos_self' (in self referential) to the 'world' basis
  Eigen::Vector3d getWorldFromSelf(const Eigen::Vector3d& pos_self) const;

  cv::Point2f imgXYFromSelf(const Eigen::Vector3d& pos_self, VirtualCameraTypeEnum cameraType = CAMERA_DEFAULT_NARROW_SMALL) const;

  /*
   * Returns the xy position expected on the screen of the point p [m]
   * throws a std::runtime_error if point is behind the camera
   */
  cv::Point2f imgXYFromWorldPosition(const cv::Point2f& p, VirtualCameraTypeEnum cameraType = CAMERA_DEFAULT_NARROW_SMALL) const;

  cv::Point2f imgXYFromWorldPosition(const Eigen::Vector3d& p, VirtualCameraTypeEnum cameraType = CAMERA_DEFAULT_NARROW_SMALL) const;

  /**
   * Returns position of the point from its field position.
   * throws a std::runtime_error if camera_field_transform is not available or if point is outside of the image.
   */
  cv::Point2f imgFromFieldPosition(const Eigen::Vector3d& p) const;

  /**
   * Return the pan,tilt position respectively on the robot basis, from xy in
   * robot basis.
   */
  static rhoban_geometry::PanTilt panTiltFromXY(const cv::Point2f& pos, double height);

  /**
   * Compute with the model the cartesian position of the
   * ball in model world frame viewed in the image at given
   * pixel.
   *
   * throw a runtime_error if corresponding ray does not intersect ball plane
   */
  Eigen::Vector3d ballInWorldFromPixel(const cv::Point2f& img_pos, VirtualCameraTypeEnum cameraType = CAMERA_DEFAULT_NARROW_SMALL) const;

  /**
   * Return the ray starting at camera source and going toward direction of img_pos
   */
  rhoban_geometry::Ray getRayInWorldFromPixel(const cv::Point2f& img_pos, VirtualCameraTypeEnum cameraType = CAMERA_DEFAULT_NARROW_SMALL) const;
  //rhoban_geometry::Ray getRayInWorldFromPixelWideangle(const cv::Point2f& img_pos) const; //[Sol] temp test

  /// Get the intersection between the ray corresponding to the pixel 'img_pos'
  /// and the horizontal plane at plane_height
  /// throw a runtime_error if corresponding ray does not intersect with the plane
  Eigen::Vector3d posInWorldFromPixel(const cv::Point2f& img_pos, double plane_height = 0, VirtualCameraTypeEnum cameraType = CAMERA_DEFAULT_NARROW_SMALL) const;


  /**
   * Return the expected radius for a ball at the given pixel.
   *
   * If the pixel is above horizon, a negative value is returned
   *
   * Note: this method is an approximation, the exact method could have 4
   * different results which are the intersection of a plane and a cone.
   * - Circle
   * - Ellipse
   * - Parabole
   * - Hyperbole
   */
  double computeBallRadiusFromPixel(const cv::Point2f& pos, VirtualCameraTypeEnum cameraType = CAMERA_DEFAULT_NARROW_SMALL) const;

  /// Distance to ground [m]
  double getHeight();

  /**
   * Pitch in degrees
   *   0 -> looking horizon
   * +90 -> looking the feet
   */
  rhoban_utils::Angle getPitch();

  /**
   * Yaw of the camera basis
   * -X -> right of the robot
   *  0 -> In front of the robot
   * +X -> left of the robot
   */
  rhoban_utils::Angle getYaw();

  /**
   * Yaw of the trunk in the world referential
   *
   */
  rhoban_utils::Angle getTrunkYawInWorld();

  rhoban_utils::TimeStamp getTimeStamp() const;
  /**
   * Return the timestamp in micro-seconds since epoch
   */
  uint64_t getTimeStampUs() const;
  /// Return the timestamp in [s]
  double getTimeStampDouble() const;

  //[Sol]
  double getPixelYtAtHorizon(double pixelX, double imgWidth, double imgHeight);
  
  //[Sol]
  int lineInfoFromPixel(const cv::Point2f &pos,
                                    float *dx, float *dy,
                                    int *px0, int *py0,
                                    int *px1, int *py1,
                                    int *px2, int *py2,
                                    int *px3, int *py3,
                                    double angularPitchError = -1.0);

  /**
   * Return the timestamp for the scheduler [s] including motor_delay
   */
  double getSchedulerTS() const;
  double getSchedulerTS(const rhoban_utils::TimeStamp& ts) const;

  /**
   * Sets the offset in micro-seconds between
   */
  void setClockOffset(int64_t new_offset);

  MoveScheduler* _moveScheduler;

  //TODO: [Sol] make _cameraModel accessible only by getter/setter
  rhoban::CameraModel _cameraModel;
  rhoban::CameraModel _cameraModelWideangleFull;
  rhoban::CameraModel _cameraModelWideangleQuarter;

  double monotonic_ts;
  uint64_t utc_ts;

  Eigen::Affine3d worldToSelf;
  Eigen::Affine3d selfToWorld;
  Eigen::Affine3d worldToCamera;
  Eigen::Affine3d cameraToWorld;
  Eigen::Affine3d cameraFromHeadBase;
  Eigen::Affine3d selfToCamera;
  
  /**
   * Depending on information source, transform between camera and field basis is not available
   */
  bool has_camera_field_transform;
  Eigen::Affine3d camera_from_field;
  Eigen::Affine3d field_from_camera;

  /**
   * Positions of the ball in field referential according to Vive
   */
  std::vector<Eigen::Vector3d> vive_balls_in_field;

  /**
   * Positions of the trackers (robots) in field referential according to Vive
   */
  std::vector<Eigen::Vector3d> vive_trackers_in_field;

  /**
   * Offset between steady_clock and system clock for the given camera state
   */
  int64_t clock_offset;

  hl_communication::FrameStatus frame_status;

  /**
   * Identifier of the video source which took the image
   */
  hl_communication::VideoSourceID source_id;

  /**
   * Currently, the motors are not properly timestamped, this allows to retrieve
   * information at the appropriate time for vision
   * unit is [ms]
   */
  static float motor_delay;


  //[Sol] returns coords(m) in robot(=self) frame from point(pix) on birdview img
  cv::Point2f robotPosFromBirdviewImg(double imgX, double imgY);  

  //[Sol] returns coords(m) in world frame from point(pix) on birdview img
  cv::Point2f worldPosFromBirdviewImg(double imgX, double imgY);

  //[Sol] to calculate birdview perspective unwarp transform we use simple hack - calculating coords of 1m x 1m square corners in front of the camera 
  //on warped wideangle img and then calculating perspective transformation
  //All this done in undistort filter in pipline
  //TODO: do calculations in camera state directly 
  void getBirdviewRotatedUnionSquareCorners(Eigen::Vector3d *flr, Eigen::Vector3d *frr, Eigen::Vector3d *blr, Eigen::Vector3d *brr);
  double getBirdviewPixelsInOneMeter() const;
  cv::Point2f getUnionSquareCenterOnBirdviewPix() const;
  cv::Size getBirdviewImageSize() const;

private:
  //[Sol]
  double birdviewPixelsInOneMeter;
  Eigen::Vector3d unionSquareCenterForBirdviewMeters; //Center of union square should be in front of the camera far enough to be in frame
  cv::Point2f unionSquareCenterOnBirdviewPix; //This union square will be projected at this coords in birdview image 
  cv::Size birdviewImageSize;

};
}  // namespace Utils
}  // namespace Vision

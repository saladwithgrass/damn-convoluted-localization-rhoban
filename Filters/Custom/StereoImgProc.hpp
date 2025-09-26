#pragma once

#include "Filters/Filter.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>



namespace Vision
{
namespace Filters
{
/// Generate a mono-canal image of the same size that the source image
/// - for each pixel, the value is the expected radius of the ball inside the image
/// In order to spare computing power, only a subset of the required value is
/// computed. Values between those points are obtained through interpolation.

struct StereoCircularObstacle {
  Eigen::Vector2d pos_in_world;
  double radius;
  double height;
};

class StereoImgProc : public Filter
{
public:
  StereoImgProc();
  virtual std::string getClassName() const override;
  virtual int expectedDependencies() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

  std::vector<StereoCircularObstacle> getDetectedObstacles() { return detectedObstacles; }

protected:
  virtual void process() override;
  virtual void setParameters() override;

private:
  void bindProperties();
  void importPropertiesFromRhIO();

  /// The number of columns where the 'exact' value is computed
  //ParamInt nbCols;
  /// The number of rows where the 'exact' value is computed
  //ParamInt nbRows;
  cv::Mat Ml, Dl, Mr, Dr;
  cv::Mat Rl, Pl, Rr, Pr;
  cv::Mat Q;
  cv::Mat mapl1, mapl2, mapr1, mapr2;
  int numberOfDisparities;  
  cv::Ptr<cv::StereoBM> bm;
  cv::Ptr<cv::StereoSGBM> sgbm;

  ros::Publisher poincloud_pub;
  ros::Publisher obstacles_bboxes_pub;
  ros::Publisher camera_fov_pub;
  ros::Publisher plane_pub;
  bool publishToROS = false;
  bool doCalibration = false;
  bool debugPauseOn = false;
  bool debugDisplayRectification = false;
  bool debugShowDisparity = false;
  int calibrationImgN = 0;
  cv::Mat oldImg;

  void fitRansacPlane(const std::vector<cv::Vec3f> &points, cv::Vec3f *normal, cv::Vec3f *origin, cv::Vec3f *best_plane_centroid);

  cv::Mat obstaclesAndFreeSpacesHeightMap;
  void publishCameraFOV();

  float minObstacleHeight = 0.40;
  float maxObstacleHeight = 0.60;
  float minObstacleDistFromRobot = 0.3; // Obstacle closer to the robot than this distance will be ignored
  float maxObstacleDistFromRobot = 2.0; // Obstacle farther from the robot than this distance will be ignored

  float obstacle3DClusteringClusterTolerance = 0.05;  // 5cm
  float obstacle3DClusteringMinClusterSize = 100;
  float obstacle3DClusteringMaxClusterSize = 25000;
  
  std::vector<StereoCircularObstacle> detectedObstacles;

  int pattern_size_w = 9;
  int pattern_size_h = 8;


  //tf::TransformBroadcaster br;



};

}  // namespace Filters
}  // namespace Vision

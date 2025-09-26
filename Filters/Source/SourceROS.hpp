#pragma once

#include "Filters/Source/Source.hpp"

#include "rhoban_utils/timing/time_stamp.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cstdlib>

#include <webots_ros/set_float.h>
#include <webots_ros/set_int.h>

using namespace std;

namespace Vision {
namespace Filters {
/**
 * SourceROS
 *

 */
class SourceROS : public Source {
 public:
  SourceROS();
  SourceROS(const std::string& source_name);
  virtual ~SourceROS();

  std::string getClassName() const override;
  int expectedDependencies() const override;

  Type getType() const override;

  void setRobotName(const std::string& robot_name);

  //Mats for distortion maps
  cv::Mat map1;
  cv::Mat map2;

 protected:
  void process() override;
  /**
   * Connect to camera and start capture
   */
  void startCamera();

  /**
   * End capture and close connection to camera
   */
  void endCamera();

  /**
   * Estimates the delay between the camera internal clock and the local pc clock.
   */
  // double getFrameRate() const override;

 protected:
  std::string robotName = "";
  std::string deviceName = "";
  int samplingPeriod = 20;
  double shutter;
  double gain;
  double framerate = 0;
  double whitebalance_blue;
  double whitebalance_red;

  /// Is the image stream starrted
  bool is_capturing = false;
  bool init_in_progress = false;

  image_transport::Subscriber sub;

  void imageCallback(const sensor_msgs::ImageConstPtr& msg);

  /// Last retrieval success in 'computer' time
  ::rhoban_utils::TimeStamp last_retrieval_success;

  /**
   * Measure of the time spent since the last synch occured
   */
  unsigned int elapsed_from_synch_ms;

  /**
   * Delta between the pc timestamp and the camera timestamp
   */
  double ts_delta;

  /**
   * Usefull to do stuff the first time only
   */
  bool first_run = true;

  /**
   * Previous normalized timestamp, used to verify timing errors.
   */
  ::rhoban_utils::TimeStamp last_ts;

  //Hardcoded values for pre-distortion of webots camera
  //double k1 = -0.2;
  double k1 = 0.2;
  //double k2 = 0.4;
  double k2 = 0.2;
  double k3 = 0;
  //For 1440x1080 frame
  //double fx = 800;
  //double fy = 800;  

  //For 720x540 frame
  double fx = 400;
  double fy = 400;  

  double k1_old = 0;
  double k2_old = 0;
  double k3_old = 0;
  double fx_old = 0;
  double fy_old = 0;
  double zoom_old = 0;
  void generateDistortMap();
};

}  // namespace Filters
}  // namespace Vision

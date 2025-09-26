#pragma once

#include "Filters/Source/Source.hpp"

#include "rhoban_utils/timing/time_stamp.h"

//#include <opencv2/opencv.hpp>
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"   

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>


using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

namespace Vision
{
namespace Filters
{
/**
 * SourceSpinnaker
 *

 */
class SourceSpinnaker : public Source
{
public:
  SourceSpinnaker();
  virtual ~SourceSpinnaker();

  std::string getClassName() const override;
  int expectedDependencies() const override;

  void fromJson(const Json::Value& v, const std::string& dir_name) override;
  Json::Value toJson() const override;

  Type getType() const override;

  int PrintDeviceInfo(INodeMap& nodeMap);
  int SetAcquisitionModeContinuous(INodeMap& nodeMap);


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

  /// Bind properties and monitoring variables to RhIO
  void bindProperties();
  void importPropertiesFromRhIO();
  int applyWishedProperties();
  int ConfigureTrigger(INodeMap& nodeMap, bool is_master);

  /**
   * Estimates the delay between the camera internal clock and the local pc clock.
   */
  double measureTimestampDelta();
  double getFrameRate() const override;  

private:
  std::string device_name;
  double shutter;
  double gain;
  double framerate = 0;
  double whitebalance_blue;
  double whitebalance_red;  
  std::string camera_serial;  
  std::string cameraDebugName;  
  bool is_right;
  bool is_stereo;

  int applyPropertiesDivider = 0;
  
  /// Is the image stream starrted
  bool is_capturing = false;
  bool init_in_progress = false;
  bool ros_inited = false;

  SystemPtr system  = nullptr;
  CameraPtr pCam = nullptr; 

  image_transport::Publisher pub_img;
  ros::Publisher pub_camerainfo;

  void* pMemBuffer;
  uint64_t memBufferTotalSize = 1440*1080*4; //Buffer size should be only enough for one frame (+ chunk data) to drop all old frames except recent one when vision is suspended

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

  bool dirtyHackFakeSingleImage = false;

  /**
   * Previous normalized timestamp, used to verify timing errors.
   */
  ::rhoban_utils::TimeStamp last_ts;  

  //Stereo calibration parameters
  std::vector<double> Mvec[2], Dvec[2];
  std::vector<double> Rvec[2], Pvec[2];

  bool publishToROS = false;
  bool stereo_calibration_json_present;




};

}  // namespace Filters
}  // namespace Vision

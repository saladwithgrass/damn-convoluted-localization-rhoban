#pragma once

#include "Filters/Source/SourceROS.hpp"

#include "rhoban_utils/timing/time_stamp.h"

#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cstdlib>
#include "webots_controller/Camera.hpp"

using namespace std;

namespace Vision {
namespace Filters {
/**
 * SourceWebots
 *

 */
class SourceWebots : public SourceROS {
 public:
  SourceWebots();
  ~SourceWebots();

  std::string getClassName() const override;
  int expectedDependencies() const override;

  void fromJson(const Json::Value& v, const std::string& dir_name) override;
  Json::Value toJson() const override;

  Type getType() const override;

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

  /**
   * Estimates the delay between the camera internal clock and the local pc clock.
   */
  // double getFrameRate() const override;

 private:
  WebotsDevices::Camera* camera_ptr;
};

}  // namespace Filters
}  // namespace Vision

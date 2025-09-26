#include "SourceWebots.hpp"
#include "Filters/Pipeline.hpp"
#include <rhoban_utils/util.h>
#include "RhIO.hpp"
#include "rhoban_utils/timing/benchmark.h"
#include <rhoban_utils/logging/logger.h>

#include <json/json.h>

namespace Vision {
namespace Filters {

SourceWebots::SourceWebots() : SourceROS("SourceWebots") {}

SourceWebots::~SourceWebots() { delete camera_ptr; }

Source::Type SourceWebots::getType() const { return Type::Online; }

std::string SourceWebots::getClassName() const { return "SourceWebots"; }

int SourceWebots::expectedDependencies() const { return 0; }

void SourceWebots::startCamera() {
  if (robotName == "") {
    // ROS_ERROR("No robot name specified");
    // return;
    robotName = RhIO::Root.getStr("server/hostname");
    cout << robotName << "::GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG()" << std::endl;
  }
  cout << deviceName << "::startCamera()" << std::endl;
  init_in_progress = true;
  camera_ptr = new WebotsDevices::Camera(robotName, "left");
  if (isStereo && isRight) {
    camera_ptr = new WebotsDevices::Camera(robotName, "right");
  }
  deviceName = camera_ptr->deviceName;
  camera_ptr->enable(samplingPeriod);
  is_capturing = true;
  init_in_progress = false;

  bindProperties();
}

void SourceWebots::endCamera() {
  cout << deviceName << "::endCamera()" << std::endl;
  is_capturing = false;
  init_in_progress = false;
}
void SourceWebots::process() {
  if (!is_capturing) {
    startCamera();
  } else {
    importPropertiesFromRhIO();
    applyWishedProperties();    
  }
  if (camera_ptr->getImageMsg()) imageCallback(camera_ptr->getImageMsg());
}

void SourceWebots::fromJson(const Json::Value& v, const std::string& dir_name) {
  Source::fromJson(v, dir_name);
  rhoban_utils::tryRead(v, "shutter", &shutter);
  rhoban_utils::tryRead(v, "gain", &gain);
  rhoban_utils::tryRead(v, "framerate", &framerate);
  rhoban_utils::tryRead(v, "whitebalance_blue", &whitebalance_blue);
  rhoban_utils::tryRead(v, "whitebalance_red", &whitebalance_red);
  rhoban_utils::tryRead(v, "is_right", &isRight);
  rhoban_utils::tryRead(v, "is_stereo", &isStereo);

  std::cout << "shutter=" << shutter << std::endl;
  std::cout << "gain=" << gain << std::endl;
  std::cout << "framerate=" << framerate << std::endl;
  std::cout << "whitebalance_blue=" << whitebalance_blue << std::endl;
  std::cout << "whitebalance_red=" << whitebalance_red << std::endl;
  std::cout << "is_right=" << isRight << std::endl;
  std::cout << "is_stereo=" << isStereo << std::endl;
}

Json::Value SourceWebots::toJson() const {
  Json::Value v = Source::toJson();
  v["device_name"] = deviceName;
  return v;
}

void SourceWebots::bindProperties() {
  std::string filter_path = rhio_path + getName();

  // std::string property_path = filter_path + "/";
  RhIO::IONode& node = RhIO::Root.child(filter_path);
  node.newFloat("shutter")->defaultValue(shutter);
  node.newFloat("gain")->defaultValue(gain);
  node.newFloat("framerate")->defaultValue(framerate);
  node.newFloat("whitebalance_blue")->defaultValue(whitebalance_blue);
  node.newFloat("whitebalance_red")->defaultValue(whitebalance_red);
  node.newFloat("k1")->defaultValue(k1);
  node.newFloat("k2")->defaultValue(k2);
  node.newFloat("k3")->defaultValue(k3);
  node.newFloat("fx")->defaultValue(fx);
  node.newFloat("fy")->defaultValue(fy);  
}

void SourceWebots::importPropertiesFromRhIO() {
  std::string filter_path = rhio_path + getName();

  RhIO::IONode& node = RhIO::Root.child(filter_path);
  shutter = node.getValueFloat("shutter").value;
  gain = node.getValueFloat("gain").value;
  framerate = node.getValueFloat("framerate").value;
  whitebalance_blue = node.getValueFloat("whitebalance_blue").value;
  whitebalance_red = node.getValueFloat("whitebalance_red").value;

  k1 = node.getValueFloat("k1").value;
  k2 = node.getValueFloat("k2").value;
  k3 = node.getValueFloat("k3").value;  
  fx = node.getValueFloat("fx").value;
  fy = node.getValueFloat("fy").value;  
}

int SourceWebots::applyWishedProperties(void) { 
  if((k1 != k1_old) || (k2 != k2_old) || (k3 != k3_old) || (fx != fx_old)|| (fy != fy_old)) {
    generateDistortMap();
  }
  return 0; 
}

}  // namespace Filters
}  // namespace Vision
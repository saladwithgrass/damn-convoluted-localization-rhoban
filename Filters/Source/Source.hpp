#pragma once

#include "Filters/Filter.hpp"

#include <rhoban_utils/util.h>
#include "rhoban_utils/timing/time_stamp.h"

namespace Vision {
namespace Filters {
/// Source abstract class: allow distinction between log, online and custom sources
class Source : public Filter {
 public:
  enum Type {
    /// CameraState is updated based on low level logs
    Log,
    /// CameraState is updated based on robot status
    Online,
    /// CameraState is updated in a custom fashion
    Custom
  };

  Source(const std::string& name) : Filter(name) {}

  Source(const std::string& name, const Dependencies& dependencies) : Filter(name, dependencies) {}

  virtual Type getType() const = 0;

  virtual Utils::CameraState* buildCameraState() {
    throw std::logic_error(DEBUG_INFO + "Not implemented for class " + getClassName());
  }

  /**
   * Returns the offset in micro-seconds between
   */
  virtual int64_t getClockOffset() const;

  rhoban_utils::TimeStamp frame_ts;
  virtual double getFrameRate() const;
  bool isStereo;
  bool isRight;
};
}  // namespace Filters
}  // namespace Vision

#pragma once

#include <vector>
#include "Filters/Filter.hpp"

namespace Vision {
namespace Filters {

/**
 * ColorBounding
 *
 * Detecting pixels of a field color in a yuv Image
 */
class ColorHSVBounding : public Filter {
public:
  ColorHSVBounding() : Filter("ColorHSVBounding") {}

  virtual std::string getClassName() const override { return "ColorHSVBounding"; }

  /**
   * YUV limits for field detection. In public so they are accessible from other
   * filters.
   */
  ParamInt minH, maxH, minS, maxS, minV, maxV;

protected:
  /**
   * @Inherit
   */
  virtual void process() override;

  virtual void setParameters() override;

private:
};
}
}


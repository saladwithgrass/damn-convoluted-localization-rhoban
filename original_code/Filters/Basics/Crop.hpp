#pragma once

#include "Filters/Filter.hpp"

namespace Vision
{
namespace Filters
{
/**
 * Rescale
 *
 * Divide the width and the height of the given image by a ratio
 */
class Crop : public Filter
{
public:
  Crop() : Filter("Crop")
  {
  }

  virtual std::string getClassName() const override
  {
    return "Crop";
  }

protected:
  /**
   * @Inherit
   */
  virtual void process() override;

  virtual void setParameters() override;

private:
  ParamInt left;
  ParamInt top;
  ParamInt width;
  ParamInt height;

};
}  // namespace Filters
}  // namespace Vision

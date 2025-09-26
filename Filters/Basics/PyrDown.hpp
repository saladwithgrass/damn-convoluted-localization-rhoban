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
class PyrDown : public Filter
{
public:
  PyrDown() : Filter("PyrDown")
  {
  }

  virtual std::string getClassName() const override
  {
    return "PyrDown";
  }

protected:
  /**
   * @Inherit
   */
  virtual void process() override;

  virtual void setParameters() override;

private:


};
}  // namespace Filters
}  // namespace Vision

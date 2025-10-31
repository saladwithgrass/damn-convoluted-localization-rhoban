#pragma once

#include "Filters/Filter.hpp"
//#include "ObstacleProvider.hpp"
#include <Filters/Features/FeaturesProvider.hpp>


namespace Vision
{
namespace Filters
{
/**
 * Rescale
 *
 * Divide the width and the height of the given image by a ratio
 */
//class ObstacleByDT : public ObstacleProvider {
class ObstacleByDT : public Filter, public FeaturesProvider {
public:
  //ObstacleByDT() : ObstacleProvider("ObstacleByDT")
  ObstacleByDT();
  virtual std::string getClassName() const override
  {
    return "ObstacleByDT";
  }

 virtual int expectedDependencies() const override {
    return 3;
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
  double my_fast_norm(double ax, double ay, double bx, double by);

};
}  // namespace Filters
}  // namespace Vision

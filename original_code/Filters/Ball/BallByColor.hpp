#pragma once
#include "Filters/Filter.hpp"
#include <vector>
#include "BallProvider.hpp"
#include <Filters/Features/FeaturesProvider.hpp>

namespace Vision {
namespace Filters {

/// Simplest filter for detecting the ball
/// - Takes a binary image as input and use the barycenter as a ball candidate
class BallByColor : public Filter, public FeaturesProvider{
public:
  BallByColor();

  virtual Json::Value toJson() const override;
  virtual std::string getClassName() const override;
  virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;
  virtual int expectedDependencies() const override;

protected:
  virtual void process() override;
  virtual void setParameters() override;
  double getCandidateScore(int center_x, int center_y, double radius,
                                   const cv::Mat& greenImg);
  cv::Rect_<float> getInnerPatch(int x, int y, float radius);
  cv::Rect_<float> getBoundaryPatch(int x, int y, float radius);
  double getBoundaryHalfWidth(float radius);




private:
  /// Debug Level:
  /// 0 - Silent
  /// 1 - Size and score of the roi
  ParamInt debugLevel;

  /// Minimal score for recognizing the ball
  ParamFloat scoreThreshold;
  
  ParamFloat boundaryFactor;
  ParamFloat maxBoundaryThickness;
};
}
}

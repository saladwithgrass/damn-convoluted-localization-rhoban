#pragma once

#include "Filters/Filter.hpp"

namespace Vision
{
namespace Filters
{

struct StereoObstacle {
  Eigen::Vector2d pos_in_self;
  std::vector<Eigen::Vector2d> hull_points_in_self;
  double radius;
  double height;
};

struct AccumulatedObstaclePoint {  
  Eigen::Vector3d posWorld;
  double age;
  int row, col; // In obstacleMap image
};

class ObstacleMap : public Filter
{
public:
  ObstacleMap();
  ~ObstacleMap();

  virtual std::string getClassName() const override
  {
    return "ObstacleMap";
  }
  virtual int expectedDependencies() const override
  {
    return 1;
  }
  std::vector<StereoObstacle> getDetectedObstacles() 
  {
    return detectedObstacles;
  }


private:
  /* the size of the side of the (squared) kernel
     used to compute the density */
  //ParamInt kernel_size;
  /* The resizing factor (0.5 makes the image 2 times smaller) */
  //ParamFloat resize_factor;

  std::vector<AccumulatedObstaclePoint> accumulatedObstaclesWorld;
  std::vector<StereoObstacle> detectedObstacles;

  void clusterAndFilterObstaclesByConnectedComponents(cv::Mat obstaclesMask, std::vector<std::vector<cv::Point> > &hulls, std::vector<cv::RotatedRect> &minRects);

  double ageDeltaPerFrame = 10.0;

  

protected:
  /**
   * @Inherit
   */
  virtual void process() override;
  virtual void setParameters() override;
};

}  // namespace Filters
}  // namespace Vision

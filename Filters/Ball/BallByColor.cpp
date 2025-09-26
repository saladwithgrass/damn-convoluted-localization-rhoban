#include "Filters/Ball/BallByColor.hpp"

#include "Filters/Patches/PatchProvider.hpp"
#include "Utils/RotatedRectUtils.hpp"
#include "Utils/Interface.h"
#include "Utils/OpencvUtils.h"
#include "Utils/ROITools.hpp"
#include "rhoban_utils/timing/benchmark.h"
#include "Utils/PatchTools.hpp"


#include "rhoban_geometry/circle.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <utility>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

using namespace std;
using namespace rhoban_geometry;
using ::rhoban_utils::Benchmark;

namespace Vision {
namespace Filters {

BallByColor::BallByColor() : Filter("BallByColor")
{
}


void BallByColor::setParameters()
{
  debugLevel = ParamInt(0,0,1);
  scoreThreshold = ParamFloat(0.5,0.0,1.0);
  boundaryFactor = ParamFloat(2.0, 1.0, 3.0);
  maxBoundaryThickness = ParamFloat(30.0, 1.0, 100.0);

  scoreThreshold = ParamFloat(100.0, 0.0, 510.);


  params()->define<ParamFloat>("boundaryFactor", &boundaryFactor);
  params()->define<ParamFloat>("maxBoundaryThickness", &maxBoundaryThickness);
  params()->define<ParamInt>("debugLevel", &debugLevel);
  params()->define<ParamFloat>("scoreThreshold", &scoreThreshold);
  
}

std::string BallByColor::getClassName() const {
  return "BallByColor";
}
Json::Value BallByColor::toJson() const
{
  Json::Value v = Filter::toJson();
//   v["arch_path"] = arch_path;
//   v["weights_path"] = weights_path;
  return v;
}

void BallByColor::fromJson(const Json::Value & v, const std::string & dir_name)
{
  Filter::fromJson(v, dir_name);
//   rhoban_utils::tryRead(v,"arch_path",&arch_path);
//   rhoban_utils::tryRead(v,"weights_path",&weights_path);
}

int BallByColor::expectedDependencies() const {
  return 4;
}

double BallByColor::getCandidateScore(int center_x, int center_y, double radius,
                                   const cv::Mat& greenImg)
{
  // Computing inner patches
  cv::Rect_<float> boundaryPatch = getBoundaryPatch(center_x, center_y, radius);
  cv::Rect_<float> innerPatch = getInnerPatch(center_x, center_y, radius);

  // Computing green_score
  double green_b = Utils::getPatchDensity(boundaryPatch, greenImg);
  double green_i = Utils::getPatchDensity(innerPatch, greenImg);
  double green_score = green_b - green_i;
  //std::cout << "center_x = " << center_x << "center_y = " << center_y << "radius = " << radius << "green_score = " << green_score << "y_score_far = " << y_score_far << "y_score_close = " << y_score_close <<std::endl; 
  return green_score;
}

cv::Rect_<float> BallByColor::getBoundaryPatch(int x, int y, float radius)
{
  // Creating boundary patch
  cv::Point2f center(x, y);
  double halfWidth = getBoundaryHalfWidth(radius);
  cv::Point2f halfSize(halfWidth, halfWidth);
  return cv::Rect_<float>(center - halfSize, center + halfSize);
}

cv::Rect_<float> BallByColor::getInnerPatch(int x, int y, float radius)
{
  // Creating inner patch
  cv::Point2f center(x, y);
  cv::Point2f halfSize(radius, radius);
  return cv::Rect_<float>(center - halfSize, center + halfSize);
}

double BallByColor::getBoundaryHalfWidth(float radius)
{
  // return radius * boundaryFactor;
  return std::min(radius * boundaryFactor, radius + maxBoundaryThickness);
}


void BallByColor::process() {

  // clearBallsData();
  // clearAllFeatures();
  clearBalls();
  cv::Mat output;
  std::string fitered_img_name = _dependencies[0];
  cv::Mat filtered_img = (getDependency(fitered_img_name).getImg())->clone();

  std::string green_img_name = _dependencies[1];
  cv::Mat green_img = (getDependency(green_img_name).getImg())->clone();

  std::string ball_radius_name = _dependencies[2];
  cv::Mat ball_radius = (getDependency(ball_radius_name).getImg())->clone();

  cv::Mat source;
  if (debugLevel == 1)
    source = (getDependency("sourceRaw").getImg())->clone();

  cv::erode(filtered_img, filtered_img, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);
  cv::erode(filtered_img, filtered_img, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);
  
  cv::dilate(filtered_img, filtered_img, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);
  cv::dilate(filtered_img, filtered_img, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);

//   cv::Mat imgToConnect = filtered_img & green_img;
  cv::Mat labelImage(filtered_img.size(), CV_32S);
  cv::Mat stats;
  cv::Mat centroids;
  int nLabels = connectedComponentsWithStats(filtered_img, labelImage, stats, centroids, 8);

  double bestScore = -1.0;
  int bestIndex = -1;
  for(int i=0; i<stats.rows; i++)
  {
    int x = stats.at<int>(cv::Point(0, i));
    int y = stats.at<int>(cv::Point(1, i));
    int w = stats.at<int>(cv::Point(2, i));
    int h = stats.at<int>(cv::Point(3, i));
    int x_center = x + w/2;
    int y_center = y + h/2;

    float radius = ball_radius.at<float>(y_center, x_center) * 1.1;

    if (radius < 4 || radius > 100)
    {
      continue;
    }

    if (std::abs(radius - (w + h) / 4) < 100)
    {
      double score = -1;
      try
      {
        score = getCandidateScore(x_center, y_center, radius, green_img);
      }
      catch (const std::runtime_error& exc)
      {
        printf("Failed to get score for patch at: %d,%d with radius %f: ignoring candidate \n",
                    x_center, y_center, radius);
        continue;
      }

      if (score > bestScore)
      {
          bestIndex = i;
          bestScore = score;
      }
    }
  }

  if (bestScore < scoreThreshold)
  {
    std::cout << "No ball on the img" << std::endl;
  }
  else
  {
    int x = stats.at<int>(cv::Point(0, bestIndex));
    int y = stats.at<int>(cv::Point(1, bestIndex));
    int w = stats.at<int>(cv::Point(2, bestIndex));
    int h = stats.at<int>(cv::Point(3, bestIndex));
    float x_center = x + w/2.0;
    float y_center = y + h/2.0;
    cv::Point2f ball(x_center, y_center);
    pushBall(ball);
    if (debugLevel == 1)
    {
      cv::Rect rect(x, y, w, h);
      cv::Scalar color(0, 0, 255);
      cv::rectangle(source, rect, color, 5);
    }
      
  }
  if (debugLevel == 1)
    img() = source;
}
}
}

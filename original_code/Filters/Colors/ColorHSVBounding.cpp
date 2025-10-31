#include "Filters/Filter.hpp"
#include "Filters/Colors/ColorHSVBounding.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Filters/Pipeline.hpp"
#include "Filters/Custom/WhiteLines.hpp"
#include "CameraState/CameraState.hpp"
#include <rhoban_utils/util.h>
#include "rhoban_utils/timing/benchmark.h"
#include <rhoban_utils/logging/logger.h>

namespace Vision {
namespace Filters {

void ColorHSVBounding::setParameters() {
  minH = ParamInt(124, 0, 255, ParameterType::PARAM);
  maxH = ParamInt(124, 0, 255, ParameterType::PARAM);
  minS = ParamInt(124, 0, 255, ParameterType::PARAM);
  maxS = ParamInt(124, 0, 255, ParameterType::PARAM);
  minV = ParamInt(124, 0, 255, ParameterType::PARAM);
  maxV = ParamInt(124, 0, 255, ParameterType::PARAM);

  params()->define<ParamInt>("minH", &minH);
  params()->define<ParamInt>("maxH", &maxH);
  params()->define<ParamInt>("minS", &minS);
  params()->define<ParamInt>("maxS", &maxS);
  params()->define<ParamInt>("minV", &minV);
  params()->define<ParamInt>("maxV", &maxV);
}

void ColorHSVBounding::process() {
  cv::Mat src = *(getDependency().getImg());

  cv::Mat temp(src.size(), 0, cv::Scalar(0)); // Unsigned char mat

  cv::Mat img_hsv;
  cv::Mat mb;

  Benchmark::open("cvtColor");
  cvtColor(src, img_hsv, CV_RGB2HSV);
  /*cv::Mat src_small, img_hsv_small;
  cv::pyrDown(src, src_small);  
  cvtColor(src_small, img_hsv_small, CV_RGB2HSV);
  cv::pyrUp(img_hsv_small, img_hsv);*/
  Benchmark::close("cvtColor");

  Benchmark::open("inRange");
  cv::Scalar lowLimit = cv::Scalar(minH, minS, minV);
  cv::Scalar highLimit = cv::Scalar(maxH, maxS, maxV);
  inRange(img_hsv, lowLimit, highLimit, temp);
  Benchmark::close("inRange");

  img() = temp;
}
}
}


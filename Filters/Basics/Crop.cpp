#include "Filters/Basics/Crop.hpp"

#include <opencv2/imgproc/imgproc.hpp>

namespace Vision
{
namespace Filters
{
void Crop::setParameters()
{
  left = ParamInt(0, 0, 10000, ParameterType::PARAM);
  top = ParamInt(0, 0, 10000, ParameterType::PARAM);
  width = ParamInt(0, 0, 10000, ParameterType::PARAM);
  height = ParamInt(0, 0, 10000, ParameterType::PARAM);
  params()->define<ParamInt>("left", &left);
  params()->define<ParamInt>("top", &top);
  params()->define<ParamInt>("width", &width);
  params()->define<ParamInt>("height", &height);
}

void Crop::process()
{
  cv::Mat src = *(getDependency().getImg());
  if(left+width > src.cols) {
    std::cout << "Warning: crop size bigger than input image size, returning empty image" << std::endl;
    return;
  }
  if(top+height > src.cols) {
    std::cout << "Warning: crop size bigger than input image size, returning empty image" << std::endl;    
    return;
  }
  cv::Mat cropped = src(cv::Rect(left, top, width, height));
  img() = cropped.clone();
  //std::cout << "left=" << left << ", top=" << top << ", width=" << width << ", height=" << height << std::endl;
}
}  // namespace Filters
}  // namespace Vision

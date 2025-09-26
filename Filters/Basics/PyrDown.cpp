#include "Filters/Basics/PyrDown.hpp"

#include <opencv2/imgproc/imgproc.hpp>

namespace Vision
{
namespace Filters
{
void PyrDown::setParameters()
{
  /*left = ParamInt(0, 0, 10000, ParameterType::PARAM);
  top = ParamInt(0, 0, 10000, ParameterType::PARAM);
  width = ParamInt(0, 0, 10000, ParameterType::PARAM);
  height = ParamInt(0, 0, 10000, ParameterType::PARAM);
  params()->define<ParamInt>("left", &left);
  params()->define<ParamInt>("top", &top);
  params()->define<ParamInt>("width", &width);
  params()->define<ParamInt>("height", &height);*/
}

void PyrDown::process()
{
  cv::Mat src = *(getDependency().getImg());
  cv::pyrDown(src, img());
}
}  // namespace Filters
}  // namespace Vision

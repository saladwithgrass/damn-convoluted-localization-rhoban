#include "CustomFactory.hpp"

#include "BallRadiusProvider.hpp"
#include "Recorder.hpp"
#include "ColorDensity.hpp"
//#include "ROIBasedGTP.hpp"
#include "WhiteLines.hpp"
#include "EverythingByDNN.hpp"
#include "../FilterFactory.hpp"
#include "WhiteLinesBirdview.hpp"
#include "StereoImgProc.hpp"
#include "ObstacleMap.hpp"
#include "YoloV5OpenVino.hpp"

namespace Vision
{
namespace Filters
{
void registerCustomFilters(FilterFactory* ff)
{
  ff->registerBuilder("BallRadiusProvider", []() { return std::unique_ptr<Filter>(new BallRadiusProvider); });
  ff->registerBuilder("Recorder", []() { return std::unique_ptr<Filter>(new Recorder()); });
  ff->registerBuilder("ColorDensity", []() { return std::unique_ptr<Filter>(new ColorDensity()); });
 // ff->registerBuilder("ROIBasedGTP", []() { return std::unique_ptr<Filter>(new ROIBasedGTP()); });
  ff->registerBuilder("WhiteLines", []() { return std::unique_ptr<Filter>(new WhiteLines()); });
  ff->registerBuilder("EverythingByDNN", []() { return std::unique_ptr<Filter>(new EverythingByDNN()); });
  ff->registerBuilder("YoloV5OpenVino", []() { return std::unique_ptr<Filter>(new YoloV5OpenVino()); });
  ff->registerBuilder("WhiteLinesBirdview", []() { return std::unique_ptr<Filter>(new WhiteLinesBirdview()); });
  ff->registerBuilder("StereoImgProc", []() { return std::unique_ptr<Filter>(new StereoImgProc()); });
  ff->registerBuilder("ObstacleMap", []() { return std::unique_ptr<Filter>(new ObstacleMap()); });

}
}  // namespace Filters
}  // namespace Vision

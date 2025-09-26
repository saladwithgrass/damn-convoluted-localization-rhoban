#include "FeaturesFactory.hpp"

#include "TagsDetector.hpp"
#include "TagsDetectorSimpleCameraCalibration.hpp"

#include "../FilterFactory.hpp"

namespace Vision
{
namespace Filters
{
void registerFeaturesFilters(FilterFactory* ff)
{
  ff->registerBuilder("TagsDetector", []() { return std::unique_ptr<Filter>(new TagsDetector); });
  ff->registerBuilder("TagsDetectorSimpleCameraCalibration", []() { return std::unique_ptr<Filter>(new TagsDetectorSimpleCameraCalibration); });
}

}  // namespace Filters
}  // namespace Vision

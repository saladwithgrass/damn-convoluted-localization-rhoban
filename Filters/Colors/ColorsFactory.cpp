#include "ColorsFactory.hpp"

#include "ChannelSelector.hpp"
#include "ColorBounding.hpp"
#include "ColorConverter.hpp"
#include "ColorHSVBounding.hpp"

#include "../FilterFactory.hpp"

namespace Vision
{
namespace Filters
{
void registerColorsFilters(FilterFactory* ff)
{
  ff->registerBuilder("ColorHSVBounding"  , [](){return std::unique_ptr<Filter>(new ColorHSVBounding()  );});
  ff->registerBuilder("ChannelSelector", []() { return std::unique_ptr<Filter>(new ChannelSelector()); });
  ff->registerBuilder("ColorBounding", []() { return std::unique_ptr<Filter>(new ColorBounding()); });
  ff->registerBuilder("ColorConverter", []() { return std::unique_ptr<Filter>(new ColorConverter()); });
}

}  // namespace Filters
}  // namespace Vision

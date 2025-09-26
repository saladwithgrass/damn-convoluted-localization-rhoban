#include "BallFactory.hpp"

#include "BallByDNN.hpp"
#include "BallByII.hpp"
#include "BallByColor.hpp"
#include "../FilterFactory.hpp"

namespace Vision
{
namespace Filters
{
void registerBallFilters(FilterFactory* ff)
{
  ff->registerBuilder("BallByDNN", [](){return std::unique_ptr<Filter>(new BallByDNN());});
  ff->registerBuilder("BallByII", []() { return std::unique_ptr<Filter>(new BallByII());});
  ff->registerBuilder("BallByColor", []() { return std::unique_ptr<Filter>(new BallByColor());});

}

}  // namespace Filters
}  // namespace Vision

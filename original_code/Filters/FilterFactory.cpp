#include "FilterFactory.hpp"

#include "Goal/GoalFactory.hpp"
#include "Ball/BallFactory.hpp"
#include "Basics/BasicsFactory.hpp"
#include "Colors/ColorsFactory.hpp"
#include "Custom/CustomFactory.hpp"
#include "Features/FeaturesFactory.hpp"
#include "Patches/PatchFactory.hpp"
#include "Source/SourceFactory.hpp"
#include "Obstacles/ObstacleFactory.hpp"

#include <exception>
#include <string>
#include <vector>

namespace Vision
{
namespace Filters
{
FilterFactory::FilterFactory()
{
  registerGoalFilters(this);
  registerBallFilters(this);
  registerBasicsFilters(this);
  registerColorsFilters(this);
  registerCustomFilters(this);
  registerFeaturesFilters(this);
  registerPatchFilters(this);
  registerSourceFilters(this);
  registerObstacleFilters(this);
}

}  // namespace Filters
}  // namespace Vision

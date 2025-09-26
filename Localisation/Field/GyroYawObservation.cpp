#include "GyroYawObservation.hpp"

#include "Field/Field.hpp"

#include <robocup_referee/constants.h>

using namespace rhoban_geometry;
using namespace robocup_referee;

namespace Vision {
namespace Localisation {

/// Giving zero scores to particles is a huge issue
double GyroYawObservation::pError = 0.001;

GyroYawObservation::GyroYawObservation(rhoban_utils::Angle obsAngle) { 
  obs = obsAngle; 
  //std::cout << "============== Creating GyroYawObservation with obsAngle.getSignedValue()=" << obsAngle.getSignedValue() << std::endl;
}

double GyroYawObservation::potential(const FieldPosition &p) const {
  // Computing the absolute difference of angle between what was expected and
  // what was received;
  double absDiffAngle = fabs((p.getOrientation() - obs).getSignedValue());
  //std::cout << "======absDiffAngle=" << absDiffAngle << std::endl;
  //return sigmoidScore(absDiffAngle, maxError, pError, sigmoidOffset, sigmoidLambda);
  return getScore(absDiffAngle, 90.0, 10.0);
}

double GyroYawObservation::getMinScore() const {
  return pError; 
}

 //[Sol] taken from GoalObservation.cpp
double GyroYawObservation::getScore(double error, double maxError, double tol) const {
  if (error >= maxError) return pError;
  if (error <= tol) return 1;
  return 1 - (error - tol) / (maxError - tol);
}

} //namespace Localisation
} //namespace Vision

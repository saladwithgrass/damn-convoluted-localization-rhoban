#pragma once

#include "FieldPosition.hpp"

#include "rhoban_unsorted/particle_filter/bounded_score_observation.h"
#include <rhoban_utils/util.h>

// Ensure that the particle is inside of the field

namespace Vision {
namespace Localisation {

class GyroYawObservation : public rhoban_unsorted::BoundedScoreObservation<FieldPosition> {

private:
  rhoban_utils::Angle obs;
public:
  GyroYawObservation(rhoban_utils::Angle obsAngle);

  virtual double potential(const FieldPosition &p) const;

  double getMinScore() const override;

  double getScore(double error, double maxError, double tol) const;

  /// Required because 0 score are particularly dangerous
  static double pError;

};
}
}

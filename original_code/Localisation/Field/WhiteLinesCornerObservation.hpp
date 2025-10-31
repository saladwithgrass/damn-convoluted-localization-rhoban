#pragma once

#include "Localisation/Field/SerializableFieldObservation.hpp"
#include "Filters/Custom/WhiteLinesData.hpp"

namespace Vision {
namespace Localisation {

class WhiteLinesCornerObservation : public SerializableFieldObservation {
private:

    struct SimpleFieldPosition {
      double x;
      double y;
      double angle;
    };

  Vision::Filters::WhiteLinesData brut_data;
  rhoban_utils::Angle pan;
  rhoban_utils::Angle tilt;
  double weight;

  /// [m]
  double robotHeight;

  //std::vector<FieldPosition> candidates;
  std::vector<SimpleFieldPosition> candidates; //Using full FieldPosition is too cumbersome here - we only need to store x,y,angle in it
  std::vector<double> x_candidates;
  std::vector<int> x_candidates_sign;
  std::vector<double> y_candidates;
  std::vector<int> y_candidates_sign;
  int sign_dot;
  double seg_dir;
  static bool debug;
  double distance_to_corner;
  double dist_bot_seg;

public:
  static double pError, maxAngleError, sigmoidOffset, sigmoidLambda;
  static double potential_pos50, potential_angle50;
  static double potential_exp;
  
  WhiteLinesCornerObservation();
  /**
   * panToArenaCorner angle is given in robot referential (left = +, right = -)
   */
  WhiteLinesCornerObservation(const Vision::Filters::WhiteLinesData & brut_data_,
                         const rhoban_utils::Angle &panToArenaCorner,
                         const rhoban_utils::Angle &tiltToArenaCorner,
                         double robotHeight_,
                         double weight_ = 1);

  virtual double potential(const FieldPosition &p) const;
  virtual double potential(const FieldPosition &p, bool debug) const;
  double corner_potential(const FieldPosition &p) const;
  double single_candiate_corner_potential(const SimpleFieldPosition & candidate, const FieldPosition &p) const;
  double segment_potential(const FieldPosition &p, bool debug) const; 
  
  rhoban_utils::Angle getPan() const;
  rhoban_utils::Angle getTilt() const;
  double getWeight() const;
  
  static void bindWithRhIO();
  static void importFromRhIO();

  std::string getClassName() const override;
  Json::Value toJson() const override;
  virtual std::string toStr() const override; 
  void fromJson(const Json::Value & v, const std::string & dir_name) override;

  double getMinScore() const override;
  Vision::Filters::WhiteLinesData getBrutData();

  double getScore(double error, double maxError, double tol) const;
};
}
}

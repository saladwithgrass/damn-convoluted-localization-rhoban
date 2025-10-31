#include "WhiteLinesCornerObservation.hpp"
#include <math.h>
#include "robocup_referee/constants.h"
#include "Field/Field.hpp"
#include "CameraState/CameraState.hpp"
#include "Filters/Custom/WhiteLinesData.hpp"
#include <vector>
#include "RhIO.hpp"

#include "rhoban_utils/logging/logger.h"
static rhoban_utils::Logger out("WhiteLinesCornerObservation");

using Vision::Utils::CameraState;
using robocup_referee::Constants;
using namespace rhoban_utils;

// double getMinScore() const override; à implémenter
// les observations elle naviguent entre pError et 1.0

namespace Vision {
namespace Localisation {

double WhiteLinesCornerObservation::maxAngleError = 30;
double WhiteLinesCornerObservation::sigmoidOffset = 0.6;
double WhiteLinesCornerObservation::sigmoidLambda = 5;

//static double arenaLength = Constants::field.field_length + 2*Constants::field.border_strip_width_x;
//static double arenaWidth = Constants::field.field_width + 2*Constants::field.border_strip_width_y;

  
bool WhiteLinesCornerObservation::debug = false;
//bool WhiteLinesCornerObservation::debug = true;

WhiteLinesCornerObservation::WhiteLinesCornerObservation() {}

WhiteLinesCornerObservation::WhiteLinesCornerObservation(const Vision::Filters::WhiteLinesData & brut_data_,
                                               const Angle &panToArenaCorner,
                                               const Angle &tiltToArenaCorner,
                                               double robotHeight_,
                                               double weight_) {
  brut_data = brut_data_;
  pan = panToArenaCorner;
  tilt = tiltToArenaCorner;
  robotHeight = robotHeight_;
  weight = weight_;
  sign_dot = 1;
  seg_dir = 0;

  if (brut_data.hasCorner) {
    auto C = brut_data.getCornerInSelf();
    auto lines = brut_data.getLinesInSelf(); 
    auto A = lines[0].first;
    auto B = lines[1].second;

    //Creating corner coordinate system consists of e1 and e2 orthogonal vectors
    cv::Point2f e1 = A-C;
    double e1_len = cv::norm(e1);
    if (e1_len==0)
      throw std::string("WhiteLinesCornerObservation e1_len null");
    e1 = (1.0 / e1_len) * e1;
    cv::Point2f e2(-e1.y, e1.x); //make e2 orthogonal to e1
    cv::Point2f bot_in_corner_frm(e1.dot(-C), e2.dot(-C)); //position of robot in corner's coordinate system 
    if (bot_in_corner_frm.x == 0 && bot_in_corner_frm.y == 0)
      throw std::string("WhiteLinesCornerObservation - robot directly on the corner (TODO)");
    double bot_arg_in_corner_frm = atan2(bot_in_corner_frm.y, bot_in_corner_frm.x);
    double corner_arg_in_bot = atan2(C.y, C.x);
    Angle bot_orient = Angle(rad2deg(bot_arg_in_corner_frm + M_PI - corner_arg_in_bot));
    distance_to_corner = sqrt(pow(bot_in_corner_frm.x,2) + pow(bot_in_corner_frm.y, 2));

    if (debug) {
      std::ostringstream oss;
      oss << "Creating white lines corner obs: pan=" << pan.getSignedValue() << ", tilt=" << tilt.getSignedValue() <<", DistToCorner=" << distance_to_corner;
      out.log(oss.str().c_str());
    }
    
    candidates.clear();

    /*
      L-shaped corners:
                       TOP
      -----------------------------------------------
      |180                270|180                270|
      |90                    |                     0|
      |------------          |          ------------|
      |180     270|         0|90        |180     270|
      |90         |        -----        |          0|
      ---------   |      /270|180\      |   ---------
      |180 270|   |     /    Y    \     |   |180 270|
      |       |   |    |     ^     |    |   |       |
      |       |   |    |     |     |    |   |       |
    L |       |   |    | X<--*     |    |   |       | R
      |       |   |    |           |    |   |       |
      |90    0|   |     \    |    /     |   |90    0|
      ---------   |      \  0|90 /      |   --------|
      |180        |        -----        |        270|
      |90        0|       270|180       |90        0|
      |------------          |          ------------|
      |180                   |                   270|
      |90                   0|90                   0|
      ----------------------------------------------
                       BOT

      How to determine bot_in_corner_frm.x/y and sign for corner coords C:
      angle 0:   C.x + bot_in_corner_frm.x, C.y + bot_in_corner_frm.y
      angle 90:  C.x - bot_in_corner_frm.y, C.y + bot_in_corner_frm.x
      angle 180: C.x - bot_in_corner_frm.x, C.y - bot_in_corner_frm.y
      angle 270: C.x + bot_in_corner_frm.x, C.y + bot_in_corner_frm.y

    */

    //Filed corners
    SimpleFieldPosition br_corner_pos = { -Constants::field.field_length/2 + bot_in_corner_frm.x,
				                         -Constants::field.field_width/2  + bot_in_corner_frm.y,
				                          (bot_orient + Angle(0)).getSignedValue() };
    candidates.push_back(br_corner_pos);

    SimpleFieldPosition tr_corner_pos = {  Constants::field.field_length/2 - bot_in_corner_frm.y,
                                 -Constants::field.field_width/2  + bot_in_corner_frm.x,
                                  (bot_orient + Angle(90)).getSignedValue() };
    candidates.push_back(tr_corner_pos);

    SimpleFieldPosition tl_corner_pos = {  Constants::field.field_length/2 - bot_in_corner_frm.x,
                                  Constants::field.field_width/2  - bot_in_corner_frm.y,
                                  (bot_orient + Angle(180)).getSignedValue() };
    candidates.push_back(tl_corner_pos);

    SimpleFieldPosition bl_corner_pos = { -Constants::field.field_length/2 + bot_in_corner_frm.y,
                                  Constants::field.field_width/2  - bot_in_corner_frm.x,
                                  (bot_orient + Angle(270)).getSignedValue() };
    candidates.push_back(bl_corner_pos); 

    
    //Middle line corners
    SimpleFieldPosition tmr_corner_pos = {                              0 - bot_in_corner_frm.x,
                                   Constants::field.field_width/2 - bot_in_corner_frm.y,
                                   (bot_orient + Angle(180)).getSignedValue() };
    candidates.push_back(tmr_corner_pos);  

    SimpleFieldPosition tml_corner_pos = {                              0 + bot_in_corner_frm.y, 
                                   Constants::field.field_width/2 - bot_in_corner_frm.x,
                                   (bot_orient + Angle(270)).getSignedValue() };
    candidates.push_back(tml_corner_pos);      

    SimpleFieldPosition bmr_corner_pos = {                              0 - bot_in_corner_frm.y,
                                  -Constants::field.field_width/2 + bot_in_corner_frm.x,
                                   (bot_orient + Angle(90)).getSignedValue() };
    candidates.push_back(bmr_corner_pos);  
    
    SimpleFieldPosition bml_corner_pos = {                              0 + bot_in_corner_frm.x,
                                  -Constants::field.field_width/2 + bot_in_corner_frm.y,
                                   (bot_orient + Angle(0)).getSignedValue() };
    candidates.push_back(bml_corner_pos);       

    //Central circle corners - top part
    SimpleFieldPosition tcbr_corner_pos = {                              0 - bot_in_corner_frm.x,
                                   Constants::field.center_radius  - bot_in_corner_frm.y,
                                   (bot_orient + Angle(180)).getSignedValue() };
    candidates.push_back(tcbr_corner_pos); 

    SimpleFieldPosition tcbl_corner_pos = {                              0 + bot_in_corner_frm.y, 
                                   Constants::field.center_radius  - bot_in_corner_frm.x,
                                   (bot_orient + Angle(270)).getSignedValue() };
    candidates.push_back(tcbl_corner_pos);      

    SimpleFieldPosition tctr_corner_pos = {                              0 - bot_in_corner_frm.y,
                                   Constants::field.center_radius  + bot_in_corner_frm.x,
                                   (bot_orient + Angle(90)).getSignedValue() };
    candidates.push_back(tctr_corner_pos);  
    
    SimpleFieldPosition tctl_corner_pos = {                              0 + bot_in_corner_frm.x,
                                   Constants::field.center_radius  + bot_in_corner_frm.y,
                                   (bot_orient + Angle(0)).getSignedValue() };
    candidates.push_back(tctl_corner_pos);       

    //Central circle corners - bottom part
    SimpleFieldPosition bcbr_corner_pos = {                              0 - bot_in_corner_frm.x,
                                  -Constants::field.center_radius  - bot_in_corner_frm.y,
                                   (bot_orient + Angle(180)).getSignedValue() };
    candidates.push_back(bcbr_corner_pos); 

    SimpleFieldPosition bcbl_corner_pos = {                              0 + bot_in_corner_frm.y, 
                                  -Constants::field.center_radius  - bot_in_corner_frm.x,
                                   (bot_orient + Angle(270)).getSignedValue() };
    candidates.push_back(bcbl_corner_pos);      

    SimpleFieldPosition bctr_corner_pos = {                              0 - bot_in_corner_frm.y,
                                  -Constants::field.center_radius  + bot_in_corner_frm.x,
                                   (bot_orient + Angle(90)).getSignedValue() };
    candidates.push_back(bctr_corner_pos);  
    
    SimpleFieldPosition bctl_corner_pos = {                              0 + bot_in_corner_frm.x,
                                  -Constants::field.center_radius  + bot_in_corner_frm.y,
                                   (bot_orient + Angle(0)).getSignedValue() };
    candidates.push_back(bctl_corner_pos); 

    //Corners of left goal zone
    SimpleFieldPosition lgz_tlt_corner_pos = {  Constants::field.field_length/2 - bot_in_corner_frm.y, 
                                       Constants::field.goal_area_width/2  + bot_in_corner_frm.x,
                                       (bot_orient + Angle(90)).getSignedValue() };
    candidates.push_back(lgz_tlt_corner_pos); 
    SimpleFieldPosition lgz_tlb_corner_pos = {  Constants::field.field_length/2 - bot_in_corner_frm.x, 
                                       Constants::field.goal_area_width/2  - bot_in_corner_frm.y,
                                       (bot_orient + Angle(180)).getSignedValue() };
    candidates.push_back(lgz_tlb_corner_pos);     
    SimpleFieldPosition lgz_blt_corner_pos = {  Constants::field.field_length/2 - bot_in_corner_frm.y, 
                                      -Constants::field.goal_area_width/2  + bot_in_corner_frm.x,
                                       (bot_orient + Angle(90)).getSignedValue() };
    candidates.push_back(lgz_blt_corner_pos); 
    SimpleFieldPosition lgz_blb_corner_pos = {  Constants::field.field_length/2 - bot_in_corner_frm.x, 
                                      -Constants::field.goal_area_width/2  - bot_in_corner_frm.y,
                                       (bot_orient + Angle(180)).getSignedValue() };
    candidates.push_back(lgz_blb_corner_pos); 

    SimpleFieldPosition lgz_tr_corner_pos = {   Constants::field.field_length/2 - Constants::field.goal_area_length + bot_in_corner_frm.y, 
                                       Constants::field.goal_area_width/2  - bot_in_corner_frm.x,
                                       (bot_orient + Angle(270)).getSignedValue() };
    candidates.push_back(lgz_tr_corner_pos);     
    SimpleFieldPosition lgz_br_corner_pos = {   Constants::field.field_length/2 - Constants::field.goal_area_length + bot_in_corner_frm.x, 
                                      -Constants::field.goal_area_width/2  + bot_in_corner_frm.y,
                                       (bot_orient + Angle(0)).getSignedValue() };
    candidates.push_back(lgz_br_corner_pos);      

    //Corners of right goal zone
    SimpleFieldPosition rgz_trt_corner_pos = { -Constants::field.field_length/2 + bot_in_corner_frm.x, 
                                       Constants::field.goal_area_width/2  + bot_in_corner_frm.y,
                                       (bot_orient + Angle(0)).getSignedValue() };
    candidates.push_back(rgz_trt_corner_pos); 
    SimpleFieldPosition rgz_trb_corner_pos = { -Constants::field.field_length/2 + bot_in_corner_frm.y, 
                                       Constants::field.goal_area_width/2  - bot_in_corner_frm.x,
                                       (bot_orient + Angle(270)).getSignedValue() };
    candidates.push_back(rgz_trb_corner_pos);     
    SimpleFieldPosition rgz_brt_corner_pos = { -Constants::field.field_length/2 + bot_in_corner_frm.x, 
                                      -Constants::field.goal_area_width/2  + bot_in_corner_frm.y,
                                       (bot_orient + Angle(0)).getSignedValue() };
    candidates.push_back(rgz_brt_corner_pos); 
    SimpleFieldPosition rgz_brb_corner_pos = { -Constants::field.field_length/2 + bot_in_corner_frm.y, 
                                      -Constants::field.goal_area_width/2  - bot_in_corner_frm.x,
                                       (bot_orient + Angle(270)).getSignedValue() };
    candidates.push_back(rgz_brb_corner_pos); 

    SimpleFieldPosition rgz_tl_corner_pos = {  -(Constants::field.field_length/2 - Constants::field.goal_area_length) - bot_in_corner_frm.x, 
                                       Constants::field.goal_area_width/2 - bot_in_corner_frm.y,
                                       (bot_orient + Angle(180)).getSignedValue() };
    candidates.push_back(rgz_tl_corner_pos);     
    SimpleFieldPosition rgz_bl_corner_pos = {  -(Constants::field.field_length/2 - Constants::field.goal_area_length) - bot_in_corner_frm.y, 
                                      -Constants::field.goal_area_width/2  + bot_in_corner_frm.x,
                                       (bot_orient + Angle(90)).getSignedValue() };
    candidates.push_back(rgz_bl_corner_pos);    

    //Corners of  penalty area (starting from robocup 2020 rules)
    //Will do nothing when penalty_area_width not present in field.json (for backwards compatibility with 2019 rules)
    if(Constants::field.penalty_area_width>0.1) {
      //Corners of left penalty area
      SimpleFieldPosition lpz_tlt_corner_pos = {  Constants::field.field_length/2 - bot_in_corner_frm.y, 
                                        Constants::field.penalty_area_width/2  + bot_in_corner_frm.x,
                                        (bot_orient + Angle(90)).getSignedValue() };
      candidates.push_back(lpz_tlt_corner_pos); 
      SimpleFieldPosition lpz_tlb_corner_pos = {  Constants::field.field_length/2 - bot_in_corner_frm.x, 
                                        Constants::field.penalty_area_width/2  - bot_in_corner_frm.y,
                                        (bot_orient + Angle(180)).getSignedValue() };
      candidates.push_back(lpz_tlb_corner_pos);     
      SimpleFieldPosition lpz_blt_corner_pos = {  Constants::field.field_length/2 - bot_in_corner_frm.y, 
                                        -Constants::field.penalty_area_width/2  + bot_in_corner_frm.x,
                                        (bot_orient + Angle(90)).getSignedValue() };
      candidates.push_back(lpz_blt_corner_pos); 
      SimpleFieldPosition lpz_blb_corner_pos = {  Constants::field.field_length/2 - bot_in_corner_frm.x, 
                                        -Constants::field.penalty_area_width/2  - bot_in_corner_frm.y,
                                        (bot_orient + Angle(180)).getSignedValue() };
      candidates.push_back(lpz_blb_corner_pos); 
      SimpleFieldPosition lpz_tr_corner_pos = {   Constants::field.field_length/2 - Constants::field.penalty_area_length + bot_in_corner_frm.y, 
                                        Constants::field.penalty_area_width/2  - bot_in_corner_frm.x,
                                        (bot_orient + Angle(270)).getSignedValue() };
      candidates.push_back(lpz_tr_corner_pos);     
      SimpleFieldPosition lpz_br_corner_pos = {   Constants::field.field_length/2 - Constants::field.penalty_area_length + bot_in_corner_frm.x, 
                                        -Constants::field.penalty_area_width/2  + bot_in_corner_frm.y,
                                        (bot_orient + Angle(0)).getSignedValue() };
      candidates.push_back(lpz_br_corner_pos);    


      //Corners of right penalty area
      SimpleFieldPosition rpz_trt_corner_pos = { -Constants::field.field_length/2 + bot_in_corner_frm.x, 
                                        Constants::field.penalty_area_width/2  + bot_in_corner_frm.y,
                                        (bot_orient + Angle(0)).getSignedValue() };
      candidates.push_back(rpz_trt_corner_pos); 
      SimpleFieldPosition rpz_trb_corner_pos = { -Constants::field.field_length/2 + bot_in_corner_frm.y, 
                                        Constants::field.penalty_area_width/2  - bot_in_corner_frm.x,
                                        (bot_orient + Angle(270)).getSignedValue() };
      candidates.push_back(rpz_trb_corner_pos);     
      SimpleFieldPosition rpz_brt_corner_pos = { -Constants::field.field_length/2 + bot_in_corner_frm.x, 
                                        -Constants::field.penalty_area_width/2  + bot_in_corner_frm.y,
                                        (bot_orient + Angle(0)).getSignedValue() };
      candidates.push_back(rpz_brt_corner_pos); 
      SimpleFieldPosition rpz_brb_corner_pos = { -Constants::field.field_length/2 + bot_in_corner_frm.y, 
                                        -Constants::field.penalty_area_width/2  - bot_in_corner_frm.x,
                                        (bot_orient + Angle(270)).getSignedValue() };
      candidates.push_back(rpz_brb_corner_pos); 
      SimpleFieldPosition rpz_tl_corner_pos = {  -(Constants::field.field_length/2 - Constants::field.penalty_area_length) - bot_in_corner_frm.x, 
                                        Constants::field.penalty_area_width/2 - bot_in_corner_frm.y,
                                        (bot_orient + Angle(180)).getSignedValue() };
      candidates.push_back(rpz_tl_corner_pos);     
      SimpleFieldPosition rpz_bl_corner_pos = {  -(Constants::field.field_length/2 - Constants::field.penalty_area_length) - bot_in_corner_frm.y, 
                                        -Constants::field.penalty_area_width/2  + bot_in_corner_frm.x,
                                        (bot_orient + Angle(90)).getSignedValue() };
      candidates.push_back(rpz_bl_corner_pos);    
    }                   

  }
  else if (brut_data.hasSegment) {
    float default_potential = pError;
    auto seg_in_self = brut_data.getSegmentInSelf();
    cv::Point2f A = seg_in_self.first;
    cv::Point2f B = seg_in_self.second;
    cv::Point2f AB = B-A;
    double dist_AB = cv::norm(AB);
    if (dist_AB == 0)
      throw std::string("WhiteLinesCornerObservation dist_AB = 0");
    cv::Point2f AB_unit = (1.0 / dist_AB) * AB;
    cv::Point2f AB_ortho_unit(-AB_unit.y, AB_unit.x);
    double dot_bot_seg = AB_ortho_unit.dot(A);//[m]
    dist_bot_seg = fabs(dot_bot_seg);
    sign_dot = (dot_bot_seg >= 0) ? 1 : -1;
    seg_dir = rad2deg(atan2(AB.y, AB.x));

    //std::cout << "-------------- WhiteLinesCornerObservation: dist_bot_seg=" << dist_bot_seg << std::endl; //2019 debug OK
    if (debug) {
      std::ostringstream oss;
      oss << "Creating white lines segment obs: DistToSegment=" << dist_bot_seg;
      out.log(oss.str().c_str());
    }
    
    
    // -------------- Vertical (x) lines ----------------
    x_candidates.clear();

    //Left vertical line
    x_candidates.push_back(  Constants::field.field_length/2 - dist_bot_seg );
    x_candidates_sign.push_back(1); 
    x_candidates.push_back(  Constants::field.field_length/2 + dist_bot_seg );
    x_candidates_sign.push_back(-1);        

    //Right vertical line
    x_candidates.push_back( -Constants::field.field_length/2 + dist_bot_seg );
    x_candidates_sign.push_back(-1);
    x_candidates.push_back( -Constants::field.field_length/2 - dist_bot_seg );
    x_candidates_sign.push_back(1);    

    //Middle vertical line
    x_candidates.push_back( 0.0 - dist_bot_seg );
    x_candidates_sign.push_back(1);
    x_candidates.push_back( 0.0 + dist_bot_seg ); 
    x_candidates_sign.push_back(-1);

    //Left goal zone vertical line
    x_candidates.push_back(  (Constants::field.field_length/2-Constants::field.goal_area_length) - dist_bot_seg);
    x_candidates_sign.push_back(1);
    x_candidates.push_back(  (Constants::field.field_length/2-Constants::field.goal_area_length) + dist_bot_seg);
    x_candidates_sign.push_back(-1);

    //Right goal zone vertical line
    x_candidates.push_back( -(Constants::field.field_length/2-Constants::field.goal_area_length) - dist_bot_seg);
    x_candidates_sign.push_back(1);
    x_candidates.push_back( -(Constants::field.field_length/2-Constants::field.goal_area_length) + dist_bot_seg);
    x_candidates_sign.push_back(-1);
    
    //Penalty area vertical lines (starting from robocup 2020 rules)
    //Will do nothing when penalty_area_width not present in field.json (for backwards compatibility with 2019 rules)
    if(Constants::field.penalty_area_width>0.1) {
      //Left penalty area vertical line
      x_candidates.push_back(  (Constants::field.field_length/2-Constants::field.penalty_area_length) - dist_bot_seg);
      x_candidates_sign.push_back(1);
      x_candidates.push_back(  (Constants::field.field_length/2-Constants::field.penalty_area_length) + dist_bot_seg);
      x_candidates_sign.push_back(-1);
      //Right penalty area vertical line
      x_candidates.push_back( -(Constants::field.field_length/2-Constants::field.penalty_area_length) - dist_bot_seg);
      x_candidates_sign.push_back(1);
      x_candidates.push_back( -(Constants::field.field_length/2-Constants::field.penalty_area_length) + dist_bot_seg);
      x_candidates_sign.push_back(-1);      
    }

    //-------------- Horizontal (y) lines ----------------
    y_candidates.clear();
    
    //Top horisontal line
    y_candidates.push_back(  Constants::field.field_width/2 - dist_bot_seg );
    y_candidates_sign.push_back(1);
    y_candidates.push_back( -Constants::field.field_width/2 + dist_bot_seg ); 
    y_candidates_sign.push_back(-1);

    //Bottom horisontal line
    y_candidates.push_back(  Constants::field.field_width/2 + dist_bot_seg );
    y_candidates_sign.push_back(-1);
    y_candidates.push_back( -Constants::field.field_width/2 - dist_bot_seg ); 
    y_candidates_sign.push_back(1); 

    //Penalty area horizontal lines (starting from robocup 2020 rules)
    //Will do nothing when penalty_area_width not present in field.json (for backwards compatibility with 2019 rules)
    if(Constants::field.penalty_area_width>0.1) {  
      //Top horisontal penalty area line
      y_candidates.push_back(  Constants::field.penalty_area_width/2 - dist_bot_seg );
      y_candidates_sign.push_back(1);
      y_candidates.push_back( -Constants::field.penalty_area_width/2 + dist_bot_seg ); 
      y_candidates_sign.push_back(-1);

      //Bottom horisontal penalt yarea line
      y_candidates.push_back(  Constants::field.penalty_area_width/2 + dist_bot_seg );
      y_candidates_sign.push_back(-1);
      y_candidates.push_back( -Constants::field.penalty_area_width/2 - dist_bot_seg ); 
      y_candidates_sign.push_back(1);         
    }  

  }
  else {
    throw std::string("WhiteLinesCornerObservation no corner neither segment");
  }
}

Angle WhiteLinesCornerObservation::getPan() const { return pan; }
Angle WhiteLinesCornerObservation::getTilt() const { return tilt; }
double WhiteLinesCornerObservation::getWeight() const { return weight; }

Vision::Filters::WhiteLinesData WhiteLinesCornerObservation::getBrutData() {
  return brut_data;
}

double WhiteLinesCornerObservation::pError = 0.2; //0.2;

//[Sol] taken from GoalObservation.cpp, see example graph below

/*
Potential graph for getScore:
^
|           |-tolerance-|
1.0         /^^^^^^^^^^^\
|          /             \
0.0_______/               \_______
          |-----margin------|
*/

double WhiteLinesCornerObservation::getScore(double error, double margin, double tolerance) const {
  if (error >= margin) return 0; 
  if (error <= tolerance) return 1;
  return 1 - (error - tolerance) / (margin - tolerance);
}


double WhiteLinesCornerObservation::single_candiate_corner_potential(const SimpleFieldPosition & candidate, const FieldPosition & p) const 
{
  double pos_error = (p.getRobotPosition() - rhoban_geometry::Point(candidate.x, candidate.y) ).getLength(); //m
  double angle_error = fabs((rhoban_utils::Angle(candidate.angle) - p.getOrientation()).getSignedValue()); //deg

  //Adjust size of a corner cluster depending to the distance of the corner
  //corner seen very close wil lhave cluster with size 0.5m, corner seen from 9 meters (length of the field) will have cluster with size 0.5m + 2.0m

  double pos_margin = 0.5 + distance_to_corner/3.0; //Birdview 
  double pos_tolerance = pos_margin*0.3;

  double angle_margin = 20.0 + distance_to_corner*5; //Angle of a corner not always estimated correctly, so use a soft margin here
  double angle_tolerance = angle_margin*0.2;
  
  //return sqrt(getScore(pos_error, pos_margin, pos_tolerance) * getScore(angle_error, angle_margin, angle_tolerance)); //Localise reasonably well with proper direction (final particle cluster will have particles with appropriate direction)
  return getScore(pos_error, pos_margin, pos_tolerance) * getScore(angle_error, angle_margin, angle_tolerance); //Removed sqrt for speedup
}

double WhiteLinesCornerObservation::corner_potential(const FieldPosition &p) const {
  // one takes the maximum potential
  // regarding all the candidates
  double best_one = 0;
  for (SimpleFieldPosition c : candidates) {
    float c_pot = single_candiate_corner_potential(c, p);
    if (c_pot > best_one) best_one = c_pot;
  }
  
  //Returning zero potential is a bad idea. "Sum of scores in particle filter is: 0" will hit if this observation is the only one and it's wrong
  //So let's trim potential to pError from bottom
  return std::max(best_one, pError);
}

double WhiteLinesCornerObservation::segment_potential(const FieldPosition &p, bool debug) const {


  //double pos_margin = 1.0 + dist_bot_seg/3.0;  
  //double pos_margin = 0.6 + dist_bot_seg/5.0;  //Birdview
  double pos_margin = 0.5 + dist_bot_seg/3.0;  //Birdview
  double pos_tolerance = pos_margin*0.2;

  //double angle_margin = 20.0 + dist_bot_seg*2; 
  double angle_margin = 20.0 + dist_bot_seg*5;  //Birdview
  double angle_tolerance = angle_margin * 0.2;

  double best_one_x = 0;
  for (int i=0;i<x_candidates.size();i++) {
    double x = x_candidates[i];
    //int sign_x = (x>=0) ? 1 : -1;
    int sign_x = x_candidates_sign[i];
    double pos_err = fabs(p.getRobotPosition().x-x);
    double pos_pot = getScore(pos_err, pos_margin, pos_tolerance);

    double base_angle = -sign_dot * sign_x * 90;
    double angle_error = fabs((Angle(base_angle - seg_dir) - p.getOrientation()).getSignedValue()); /* deg */
    double angle_pot = getScore(angle_error, angle_margin, angle_tolerance);

    //double pot = sqrt(pos_pot * angle_pot);
    double pot = pos_pot * angle_pot;  //Removed sqrt for speedup
    if (best_one_x < pot) best_one_x = pot;
  }

  double best_one_y = 0;
  for (int i=0;i<y_candidates.size();i++) {
    double y = y_candidates[i];
    //int sign_y = (y>=0) ? 1 : -1;
    int sign_y = y_candidates_sign[i];
    double pos_err = fabs(p.getRobotPosition().y-y);
    double pos_pot = getScore(pos_err, pos_margin, pos_tolerance);

    double base_angle = 90 - sign_dot * sign_y * 90;
    double angle_error = fabs((Angle(base_angle - seg_dir) - p.getOrientation()).getSignedValue()); /* deg */
    double angle_pot = getScore(angle_error, angle_margin, angle_tolerance);
    
    //double pot = sqrt(pos_pot * angle_pot);
    double pot = pos_pot * angle_pot;  //Removed sqrt for speedup
    if (best_one_y < pot) best_one_y = pot;
  }

  double pot_res = std::max(best_one_x, best_one_y);
  
  //Returning zero potential is a bad idea. "Sum of scores in particle filter is: 0" will hit if this observation is the only one and it's wrong
  //So let's trim potential to pError from bottom
  return std::max(pot_res, pError); 
  
}

std::string WhiteLinesCornerObservation::toStr() const {
  if (brut_data.hasCorner) {
    return "Corner Observed";
  }
  else if (brut_data.hasSegment) {
    return "Segment Observed";
  }
  else {
    return "Unconsistant observation";
  }
}
  
double WhiteLinesCornerObservation::potential(const FieldPosition &p) const {

  //return fabs(p.getRobotPositionCV().x)/10.0; //Dumb debug

  if (brut_data.hasCorner) {
    return corner_potential(p);
  }
  else if (brut_data.hasSegment) {
    return segment_potential(p, false);
  } 
  else {
    return pError;
  }
}

double WhiteLinesCornerObservation::potential(const FieldPosition &p, bool debug) const {

  //if(debug) std::cout << "DEBUGGGGGG" << std::endl; //debug=true only for representative partice potential, print some additional info if needed

  if (brut_data.hasCorner) {
    return corner_potential(p);
  }
  else if (brut_data.hasSegment) {
    return segment_potential(p, debug);
  } 
  else {
    return pError;
  }
}

void WhiteLinesCornerObservation::bindWithRhIO() {
  RhIO::Root.newFloat("/localisation/field/WhiteLinesCornerObservation/pError")
      ->defaultValue(pError)
      ->minimum(0.0)
      ->maximum(1.0)
      ->comment("The false positive probability");
  RhIO::Root.newFloat(
                 "/localisation/field/WhiteLinesCornerObservation/maxAngleError")
      ->defaultValue(maxAngleError)
      ->minimum(0.0)
      ->maximum(180)
      ->comment(
            "The maximum angle difference between expectation and observation");
  RhIO::Root.newFloat(
                 "/localisation/field/WhiteLinesCornerObservation/sigmoidOffset")
      ->defaultValue(sigmoidOffset)
      ->minimum(0.0)
      ->maximum(1.0)
      ->comment(
            "The value at which dScore/dx is lambda, with dx = dAngle/maxAngle");
  RhIO::Root.newFloat(
                 "/localisation/field/WhiteLinesCornerObservation/sigmoidLambda")
      ->defaultValue(sigmoidLambda)
      ->minimum(0.0)
      ->maximum(1000.0)
      ->comment("Cf. sigmoidOffset");
  RhIO::Root.newBool("/localisation/field/WhiteLinesCornerObservation/debug")
      ->defaultValue(debug)
      ->comment("Print message on observation creation");
}

void WhiteLinesCornerObservation::importFromRhIO() {
  RhIO::IONode &node =
      RhIO::Root.child("localisation/field/WhiteLinesCornerObservation");
  pError = node.getValueFloat("pError").value;
  maxAngleError = node.getValueFloat("maxAngleError").value;
  sigmoidOffset = node.getValueFloat("sigmoidOffset").value;
  sigmoidLambda = node.getValueFloat("sigmoidLambda").value;
  debug = node.getValueBool("debug").value;
}

std::string WhiteLinesCornerObservation::getClassName() const {
  return "WhiteLinesCornerObservation";
}

Json::Value WhiteLinesCornerObservation::toJson() const {
  Json::Value v;
  v["robotHeight"] = robotHeight;
  v["pan"] = pan.getSignedValue();
  v["tilt"] = tilt.getSignedValue();
  return v;
}

void WhiteLinesCornerObservation::fromJson(const Json::Value & v, const std::string & dir_name) {
  rhoban_utils::tryRead(v,"robotHeight",&robotHeight);
  rhoban_utils::tryRead(v,"pan",&pan);
  rhoban_utils::tryRead(v,"tilt",&tilt);
}

double WhiteLinesCornerObservation::getMinScore() const {
  return pError + 0.05;
}
}
}

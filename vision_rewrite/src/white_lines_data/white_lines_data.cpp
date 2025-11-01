#include "white_lines_data.hpp"
#include <opencv2/opencv.hpp>

WhiteLinesData::WhiteLinesData() {
  loc_active = true;
  reset();
}

void WhiteLinesData::computeTransformations(CameraState * cs, bool final_compute) {
    world_lines.clear();
    self_lines.clear();
    observation_valid = final_compute;
    try {
        for (size_t k = 0; k < pix_lines.size(); k++) {
            cv::Point2f A = pix_lines[k].first;
            cv::Point2f B = pix_lines[k].second;
            //std::cout << "WhiteLinesData: parsing line in pix:" << A << "-" << B << std::endl; //2019 checked, OK
            //auto pointA = cs->robotPosFromImg(A.x, A.y, 1, 1, false);
            //cv::Point2f pointA = cs->worldPosFromImg(A.x, A.y); //2019
            cv::Point2f pointA = cs->worldPosFromBirdviewImg(A.x, A.y); //2019 birdview

            //auto pointB = cs->robotPosFromImg(B.x, B.y, 1, 1, false);
            //cv::Point2f pointB = cs->worldPosFromImg(B.x, B.y); //2019
            cv::Point2f pointB = cs->worldPosFromBirdviewImg(B.x, B.y); //2019 birdview

            //std::cout << "WhiteLinesData: robotPosFromImg result:" << pointA << "-" << pointB << std::endl;  //2019 checked, OK
            world_lines.push_back(std::pair<cv::Point2f, cv::Point2f >(pointA, pointB));
            pointA = cs->getPosInSelf(pointA);
            pointB = cs->getPosInSelf(pointB);
            //std::cout << "WhiteLinesData: getPosInSelf result:" << pointA << "-" << pointB << std::endl;  //2019 checked, OK
            self_lines.push_back(std::pair<cv::Point2f, cv::Point2f >(pointA, pointB));
            //if (line_quality[k] > max_obs_score) observation_valid = false;
        }

        if (pix_lines.size() == 1) {
            define_segment(final_compute);
            return;
        }

        if (hasCorner) {
            //world_corner = cs->robotPosFromImg(pix_corner.x, pix_corner.y, 1, 1, false);
            //world_corner = cs->robotPosFromImg(pix_corner.x, pix_corner.y);
            //world_corner = cs->worldPosFromImg(pix_corner.x, pix_corner.y); //2019
            world_corner = cs->worldPosFromBirdviewImg(pix_corner.x, pix_corner.y); //2019 birdview
            self_corner = cs->getPosInSelf(world_corner);
            corner_robot_dist = cv::norm(self_corner);

            if (self_lines.size() >= 2) {
                // Le secteur angulaire est vu de haut, l'angle orienté ACB est
                // donc dans le sens trigo.
                // Calcul de l'angle par AlKashi
                cv::Point2f A = self_lines[0].first;
                cv::Point2f B = self_lines[1].second;
                float CA = cv::norm(A-self_corner);
                float CB = cv::norm(B-self_corner);
                float AB = cv::norm(B-A);
                if (final_compute) {
                    FBPRINT_DEBUG("distances : CA=%f CB=%f AB=%f dist(corner)=%f\n", CA, CB, AB, cv::norm(self_corner));
                }
                if (CA != 0 && CB != 0) {
                    float co = (CA*CA + CB*CB - AB*AB)/ (2*CA*CB);
                    if (co < -1.0) co = -1.0; 
                    if (co > 1.0) co = 1.0;
                    corner_angle = acos(co);
                    if (final_compute) {
                        FBPRINT_DEBUG("Corrected Corner angle (ACB) = %f\n", 180.0 / M_PI * corner_angle);
                    }
                    if (fabs(M_PI/2 - corner_angle) > deg2rad(tolerance_angle_corner) // TODO: parametre
                            || CA < minimal_segment_length || CB < minimal_segment_length ) {
                        if (final_compute) {
                            FBPRINT_DEBUG("Corner avoided (bad angle or segment too short)\n");
                        }
                        hasCorner = false;
                        observation_valid = false;
                    }

                    if (corner_robot_dist > max_dist_corner) {
                        if (final_compute) { 
                            FBPRINT_DEBUG("Corner avoided : it is too far : seen at %f\n", corner_robot_dist); 
                        }
                        observation_valid = false;
                    }

                    if (fabs(M_PI - corner_angle) < deg2rad(tolerance_angle_line)) {
                        define_segment(final_compute);
                        return;
                    }
                } else {
                    hasCorner = false;
                    observation_valid = false;
                    if (final_compute) { FBPRINT_DEBUG("Segment are too short\n"); }
                }
            } else {
                hasCorner = false;
                if (final_compute) { FBPRINT_DEBUG("There is only one line\n"); }
            }
        }
        else {
            if (final_compute) { FBPRINT_DEBUG(" No corner\n"); }
            observation_valid = false;
        }
    }
    catch ( const std::exception & e ) {
        // std::cerr << e.what();
        if (final_compute) { FBPRINT_DEBUG("CBB: exception during computation\n"); }
        rollback_computation();
    }
    FBPRINT_DEBUG("All is OK\n");
}

std::pair<cv::Point2f, cv::Point2f >
WhiteLinesData::getSegmentInSelf() const {
  return self_segment;
}

std::pair<cv::Point2f, cv::Point2f>
WhiteLinesData::getSegmentInWorld() const {
  return world_segment;
}
  
float WhiteLinesData::getRobotCornerDist() {
  return corner_robot_dist;
}
  
void WhiteLinesData::addPixCorner(cv::Point2f C) {
  hasCorner = true;
  pix_corner = C;
}
  
cv::Point2f WhiteLinesData::getPixCorner() const {
  return pix_corner;
}

cv::Point2f WhiteLinesData::getCornerInWorldFrame() const {
  return world_corner;
}

cv::Point2f WhiteLinesData::getCornerInSelf() const {
  return self_corner;
}

std::vector<std::pair<cv::Point2f, cv::Point2f > > WhiteLinesData::getLinesInSelf() const {
  return self_lines;
}
float WhiteLinesData::getCornerAngle() const {
  return corner_angle;
}
  
std::vector<std::pair<cv::Point2f, cv::Point2f > > WhiteLinesData::getLinesInWorldFrame() const {
  return world_lines;
}

void WhiteLinesData::clearLines() {
  pix_lines.clear();
  line_quality.clear();
  world_lines.clear();
  self_lines.clear();
  line_scores.clear();
}

void WhiteLinesData::reset() {
  observation_valid = false;
  hasSegment = false;
  hasCorner = false;
  corner_robot_dist = -1;
  corner_angle = -1;
  clearLines();
}

void WhiteLinesData::rollback_computation() {
  hasSegment = false;
  hasCorner = false;
  corner_robot_dist = -1;
  corner_angle = -1;
  world_lines.clear();
  self_lines.clear();
  observation_valid = false;
}

std::vector<std::pair<cv::Point2f, cv::Point2f > > WhiteLinesData::getPixLines() const {
  return pix_lines;
}
  
void WhiteLinesData::pushPixLine(double quality, std::pair<cv::Point2f,cv::Point2f> L) {
  if (cv::norm(L.first-L.second) > 0) {
    pix_lines.push_back(L);
    line_quality.push_back(quality);
  }
}

bool WhiteLinesData::is_obs_valid() {
  return loc_active && observation_valid;
}
  
}
}

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Eigen>
#include "camera_state.hpp"

cv::Point2f CameraState::getPosInSelf(const cv::Point2f& pos_in_origin) const {
    Eigen::Vector3d pos_in_self = worldToSelf * 
        Eigen::Vector3d(pos_in_origin.x, pos_in_origin.y, 0);
    return cv::Point2f(pos_in_self(0), pos_in_self(1));
}

cv::Point2f CameraState::worldPosFromBirdviewImg(double imgX, double imgY) {
    cv::Point2f posInSelfCV = robotPosFromBirdviewImg(imgX, imgY);
    Eigen::Vector3d posInSelf(posInSelfCV.x, posInSelfCV.y, 0);
    Eigen::Vector3d posInWorld = selfToWorld * posInSelf;

    return cv::Point2f(posInWorld(0), posInWorld(1));
}

cv::Point2f CameraState::getUnionSquareCenterOnBirdviewPix() const { return unionSquareCenterOnBirdviewPix; }

cv::Point2f CameraState::robotPosFromBirdviewImg(double imgX, double imgY) {
    // Getting coords in m in birdview frame rotated by camera yaw
    Eigen::Vector3d rotatedRobotPos;
    cv::Point2f unionSquareCenterOnBirdviewPix = getUnionSquareCenterOnBirdviewPix();
    rotatedRobotPos[0] =
        -(imgY - unionSquareCenterOnBirdviewPix.y) / birdviewPixelsInOneMeter + unionSquareCenterForBirdviewMeters[0];
    rotatedRobotPos[1] =
        -(imgX - unionSquareCenterOnBirdviewPix.x) / birdviewPixelsInOneMeter + unionSquareCenterForBirdviewMeters[1];
    rotatedRobotPos[2] = unionSquareCenterForBirdviewMeters[2];  // Shoud be zero, but for code clarity

    // Unrotate
    // double cameraYaw = getYaw().getSignedValue();
    double cameraYaw = getYaw();
    Eigen::Affine3d a = Eigen::Affine3d::Identity();
    a.linear() = Eigen::AngleAxisd(cameraYaw * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Vector3d unrotatedRobotPos = a * rotatedRobotPos;
    return cv::Point2f(unrotatedRobotPos(0), unrotatedRobotPos(1));
}

// int CameraState::lineInfoFromPixel(
//         const cv::Point2f& pos, 
//         float* dx, float* dy, 
//         int* px0, int* py0, 
//         int* px1, int* py1,
//         int* px2, int* py2, 
//         int* px3, int* py3, 
//         double angularPitchError) {
//   // Ray cameraCentralRay = getRayInWorldFromPixel(cv::Point2f(getImgSize().width/2, getImgSize().height/2));
//
//   Ray cameraCentralRay = getRayInWorldFromPixel(cv::Point2f(getImgSize().width / 2, 0));
//   // Ray cameraCentralRay = getRayInWorldFromPixel(pos);
//   // Ray consist of source and dir, both Eigen3D
//
//   Eigen::Vector3d forwardDir = Eigen::Vector3d(cameraCentralRay.dir(0), cameraCentralRay.dir(1), 0.0).normalized();
//   Eigen::Vector3d leftDir = Eigen::Vector3d(cameraCentralRay.dir(1), -cameraCentralRay.dir(0), 0.0).normalized();
//   // Eigen::Vector3d leftDir = Eigen::Vector3d(0.0, 0.0, 0.0);
//   // Eigen::Vector3d leftDir = viewRay.dir.cross(groundDir).normalized();
//
//   Ray viewRay = getRayInWorldFromPixel(pos);
//   if (viewRay.dir.z() >= 0)  // Point above horison (z componnet of direction >0)
//   {
//     return -1;
//   }
//   Plane groundPlane(Eigen::Vector3d(0, 0, 1), 0.0);
//
//   if (!isIntersectionPoint(viewRay, groundPlane)) {
//     return -1;
//   }
//
//   Eigen::Vector3d posInWorld = getIntersection(viewRay, groundPlane);
//   cv::Point2f pof = cv::Point2f(posInWorld(0), posInWorld(1));
//
//   cv::Point2f shiftxr = cv::Point2f(leftDir(0) * 0.05 / 2, leftDir(1) * 0.05 / 2);
//   cv::Point2f shiftyr = cv::Point2f(forwardDir(0) * 0.05 / 2, forwardDir(1) * 0.05 / 2);
//
//   /*
//   //Calculating vector to shift pixel to half of line width in "x" (from left to right) direction in image taking into
//   account camera Yaw cv::Point2f shiftx = cv::Point2f(0.0, 0.05/2.0); //Shift to the left cv::Point2f shiftxr =
//   cv::Point2f(shiftx.x * cos(getYaw()) - shiftx.y * sin(getYaw()), shiftx.x * sin(getYaw()) + shiftx.y * cos(getYaw())
//   );
//
//   //Calculating vector to shift pixel to half of line width in "y" (from bottom to top) direction in image taking into
//   account camera Yaw
//   //cv::Point2f shifty = cv::Point2f(0.05/2.0, 0.00); //Shift forward
//   cv::Point2f shifty = cv::Point2f(0.0, 0.00); //Shift forward
//   cv::Point2f shiftyr = cv::Point2f(shifty.x * cos(getYaw()) - shifty.y * sin(getYaw()),
//                                    shifty.x * sin(getYaw()) + shifty.y * cos(getYaw()) );
//
//   //cv::Point2f pof = robotPosFromImg(pos.x, pos.y); //Get point on field from pixel coords
//   cv::Point2f pof = worldPosFromImg(pos.x, pos.y); //Get point on field from pixel coords, 2019 version
//   */
//   cv::Point2f pim0sx, pim1sx, pim0sy, pim1sy;
//
//   try {
//     pim0sx = imgXYFromWorldPosition(pof - shiftxr);  // Get pixel coords of shifted point on field
//     pim1sx = imgXYFromWorldPosition(pof + shiftxr);  // Get pixel coords of shifted point on field
//     pim0sy = imgXYFromWorldPosition(pof - shiftyr);  // Get pixel coords of shifted point on field
//     pim1sy = imgXYFromWorldPosition(pof + shiftyr);  // Get pixel coords of shifted point on field
//   } catch (const std::runtime_error& exc) {
//     return -1;
//   }
//
//   /*
//   //cv::Point2f pim0sx = imgXYFromRobotPosition2f(pof - shiftxr, width, height); //Get pixel coords of shifted point on
//   field
//   //cv::Point2f pim1sx = imgXYFromRobotPosition2f(pof + shiftxr, width, height); //Get pixel coords of shifted point on
//   field
//   //cv::Point2f pim0sy = imgXYFromRobotPosition2f(pof - shiftyr, width, height); //Get pixel coords of shifted point on
//   field
//   //cv::Point2f pim1sy = imgXYFromRobotPosition2f(pof + shiftyr, width, height); //Get pixel coords of shifted point on
//   field
//   */
//
//   *px0 = pim0sx.x;
//   *py0 = pim0sx.y;
//   *px1 = pim1sx.x;
//   *py1 = pim1sx.y;
//   *px2 = pim0sy.x;
//   *py2 = pim0sy.y;
//   *px3 = pim1sy.x;
//   *py3 = pim1sy.y;
//
//   // *dx = fabs(pim0sx.x - pim1sx.x);
//   // *dy = fabs(pim0sy.y - pim1sy.y);
//   *dx = sqrt(pow(pim0sx.x - pim1sx.x, 2) + pow(pim0sx.y - pim1sx.y, 2));
//   *dy = sqrt(pow(pim0sy.x - pim1sy.x, 2) + pow(pim0sy.y - pim1sy.y, 2));
//
//   return 0;  // no errors
// }

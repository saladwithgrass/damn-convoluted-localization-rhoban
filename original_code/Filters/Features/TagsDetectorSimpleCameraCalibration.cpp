#include "Filters/Features/TagsDetectorSimpleCameraCalibration.hpp"

#include "rhoban_utils/logging/logger.h"
#include "rhoban_utils/timing/benchmark.h"

#include "CameraState/CameraState.hpp"

#include <vector>
#include <iostream>


/*
  [Sol] use this filter to save some images for simple OpenCV camera calibration in case when full 
  rhoban robot calibration process is too cumbersome or aruco calibration area is unaviable.

  Json usage:
    {
        "class name" : "TagsDetectorSimpleCameraCalibration",
        "content" : {
            "name" : "tagsDetectorSimpleCameraCalibration",
            "dependencies" : ["sourceRaw"]
        }
    },  

  1. Print https://github.com/abhishek098/camera_calibration/blob/master/aruco_marker_board.pdf

  2. When filter shows corectly detected aruco markers (partial occlusion is OK), 
     call "saveImage" in rhio to save jpg to /tmp folder

  3. Copy this images from robot's /tmp to your /workspace/env/[robot_name]/calibration/simpleCameraCalibrationWideFull

  4. Launch calibrate.sh there

  5. Resulting calibration.yaml will contains:

  [focal_x   0.0      center_x]
  [  0.0    focal_y   center_y]
  [  0.0     0.0         1.0  ]

  [ k1 k2 p1 p2
    k3 k4 kp k6 ], k - radial, p - tangential

  Use this values in calibration_wideangle_full.json. 
  Only this camera type calibration is required for current spiinaker_wideangle robots
  
  6. Launch this filter with "undistort=true" and view out to check the result  

*/


using namespace std;
using ::rhoban_utils::Benchmark;

static rhoban_utils::Logger out("TagsDetectorSimpleCameraCalibration");

namespace Vision
{
namespace Filters
{

TagsDetectorSimpleCameraCalibration::TagsDetectorSimpleCameraCalibration()
  : Filter("TagsDetectorSimpleCameraCalibration")
  , detectorParameters(new cv::aruco::DetectorParameters())

{
  imageN = 0;
  bindRhio();
}

void TagsDetectorSimpleCameraCalibration::setParameters()
{
  adaptiveThreshConstant = ParamFloat(7, 3, 23);
  params()->define<ParamFloat>("adaptiveThreshConstant", &adaptiveThreshConstant);
  adaptiveThreshWinSizeMin = ParamInt(7, 3, 23);
  params()->define<ParamInt>("adaptiveThreshWinSizeMin", &adaptiveThreshWinSizeMin);
  adaptiveThreshWinSizeMax = ParamInt(7, 3, 23);
  params()->define<ParamInt>("adaptiveThreshWinSizeMax", &adaptiveThreshWinSizeMax);
  adaptiveThreshWinSizeStep = ParamInt(7, 3, 23);
  params()->define<ParamInt>("adaptiveThreshWinSizeStep", &adaptiveThreshWinSizeStep);
  cornerRefinementMaxIterations = ParamInt(50, 20, 100);
  params()->define<ParamInt>("cornerRefinementMaxIterations", &cornerRefinementMaxIterations);
  cornerRefinementMinAccuracy = ParamFloat(0.01, 0.0001, 1);
  params()->define<ParamFloat>("cornerRefinementMinAccuracy", &cornerRefinementMinAccuracy);
  cornerRefinementWinSize = ParamInt(3, 1, 10);
  params()->define<ParamInt>("cornerRefinementWinSize", &cornerRefinementWinSize);

  // marker size in m
  markerSize = ParamFloat(0.09, 0, 1.0);
  params()->define<ParamFloat>("markerSize", &markerSize);

  debugLevel = ParamInt(0, 0, 2);
  params()->define<ParamInt>("debugLevel", &debugLevel);
  period = ParamInt(1, 1, 100);
  params()->define<ParamInt>("period", &period);
  isWritingData = ParamInt(0, 0, 1);
  params()->define<ParamInt>("isWritingData", &isWritingData);
  refine = ParamInt(1, 0, 1);
  params()->define<ParamInt>("refine", &refine);
}

void TagsDetectorSimpleCameraCalibration::process()
{

  updateControl();

  const cv::Mat& srcImg = *(getDependency().getImg());

  // Importing parameters from rhio
  // Adaptive threshold is used since it is note modified
  detectorParameters->adaptiveThreshConstant = adaptiveThreshConstant;
  detectorParameters->adaptiveThreshWinSizeMin = adaptiveThreshWinSizeMin;
  detectorParameters->adaptiveThreshWinSizeMax = adaptiveThreshWinSizeMax;
  detectorParameters->adaptiveThreshWinSizeStep = adaptiveThreshWinSizeStep;
  detectorParameters->doCornerRefinement = refine;
  detectorParameters->cornerRefinementMaxIterations = cornerRefinementMaxIterations;
  detectorParameters->cornerRefinementMinAccuracy = cornerRefinementMinAccuracy;
  detectorParameters->cornerRefinementWinSize = cornerRefinementWinSize;

  if(needToUndistort) {
    const rhoban::CameraModel& cameraModel = getCS().getCameraModel(Vision::Utils::CAMERA_WIDE_FULL);
    cv::Mat cameraMatrix = cameraModel.getCameraMatrix();
    cv::Mat distCoeffs = cameraModel.getDistortionCoeffs();
    std::cout << "cameraMatrix=" << cameraMatrix << std::endl;
    std::cout << "distCoeffs=" << distCoeffs << std::endl;

    cv::undistort(srcImg, img(), cameraMatrix, distCoeffs);
  } else {  
    img() = srcImg.clone();
  }
 

  if(needToSaveImage) {
    needToSaveImage = 0;
    char s[255];
    sprintf(s, "/tmp/img%03d.jpg", imageN++);
    cv::imwrite(s, srcImg);
  }

  cv::Ptr<cv::aruco::Dictionary> dic = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_1000);
  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners;
  cv::aruco::detectMarkers(img(), dic, markerCorners, markerIds, detectorParameters);
  cv::aruco::drawDetectedMarkers(img(), markerCorners, markerIds, cv::Scalar(0, 0, 255));

}


int TagsDetectorSimpleCameraCalibration::expectedDependencies() const
{
  return 1;
}

void TagsDetectorSimpleCameraCalibration::bindRhio()
{
  RhIO::Root.newCommand("tagsDetectorSimpleCameraCalibration/saveImage", "save image in /tmp",
                      [this](const std::vector<std::string>& args) -> std::string {
                        needToSaveImage = true;
                        return "Image saved";
                      }); 
}

void TagsDetectorSimpleCameraCalibration::initControl()
{
  std::string undistort_path = rhio_path + getName() + "/undistort";
  if (RhIO::Root.getValueType(undistort_path) != RhIO::ValueType::NoValue)
    return;
  RhIO::Root.newBool(undistort_path)->defaultValue(needToUndistort);
}

void TagsDetectorSimpleCameraCalibration::updateControl()
{
  initControl();
  std::string undistort_path = rhio_path + getName() + "/undistort";
  needToUndistort = RhIO::Root.getValueBool(undistort_path).value;
}  

}  // namespace Filters
}  // namespace Vision

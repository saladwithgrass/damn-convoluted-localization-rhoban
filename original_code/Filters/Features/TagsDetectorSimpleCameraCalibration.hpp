#pragma once

#include "Filters/Filter.hpp"

#include "rhoban_utils/tables/string_table.h"

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/aruco.hpp>
#include <string>

namespace Vision
{
namespace Filters
{
/**
 * TagsDetector
 *
 */
class TagsDetectorSimpleCameraCalibration : public Filter
{
public:

  TagsDetectorSimpleCameraCalibration();

  // Json stuff
  // virtual void fromJson(const Json::Value & v, const std::string & dir_name);
  // virtual Json::Value toJson() const;
  virtual std::string getClassName() const override
  {
    return "TagsDetectorSimleCameraCalibration";
  }

  virtual void setParameters() override;
  virtual int expectedDependencies() const override;

protected:
  /**
   * @Inherit
   */
  virtual void process() override;

private:
  ParamFloat adaptiveThreshConstant;
  ParamInt adaptiveThreshWinSizeMin;
  ParamInt adaptiveThreshWinSizeMax;
  ParamInt adaptiveThreshWinSizeStep;
  ParamInt cornerRefinementMaxIterations;
  ParamFloat cornerRefinementMinAccuracy;
  ParamInt cornerRefinementWinSize;

  // TODO: other parameters are available:
  // see: https://docs.opencv.org/3.1.0/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html

  /// Size of the markers in [m]
  ParamFloat markerSize;

  /// 0 -> no output and no copy of the image
  /// 1 -> Display markers and cubes
  /// 2 -> Verbose output
  ParamInt debugLevel;

  /// Since aruco tag execution is expensive, it is only executed every
  /// 'period' images
  ParamInt period;

  /// 0 -> data are not dumped
  /// 1 -> All collected data are dumped after every image
  ParamInt isWritingData;

  /// 0 -> don't use refine corner sub pix
  /// 1 -> use refine corner sub pix
  ParamInt refine;

  // Detection parameters
  cv::Ptr<cv::aruco::DetectorParameters> detectorParameters;

  void bindRhio();
  int imageN = 0;
  bool needToSaveImage = false; 
  bool needToUndistort = false;
  void initControl();
  void updateControl();
};
}  // namespace Filters
}  // namespace Vision

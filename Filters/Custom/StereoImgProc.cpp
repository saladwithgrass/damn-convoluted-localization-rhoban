#include "Filters/Custom/StereoImgProc.hpp"

#include "rhoban_utils/timing/benchmark.h"
#include "rhoban_utils/util.h"

#include "rhoban_utils/io_tools.h"
#include "rhoban_utils/util.h"
#include "rhoban_utils/serialization/json_serializable.h"


#include <limits>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include </usr/include/pcl-1.8/pcl/common/impl/common.hpp> //To get proper overloaded type of pcl::getMinMax3D
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>  
#include "tf/transform_datatypes.h"
#include <tf_conversions/tf_eigen.h>

#include <json/json.h>

#include "CameraState/CameraState.hpp"
#include "Filters/Pipeline.hpp"
#include "robocup_referee/constants.h"


using rhoban_utils::Benchmark;
using namespace cv;

#define USE_SGBM 0
#define USE_RECTIFICATION_FROM_JSON 1
#define PARALLEL_THREADS 8 //8
#define USE_FULL_3D_CLUSTERING 1

#define MISSING_Z  10000.f

//X looks forward, Y looks left, Z looks up

//This is a size and resolution of voxel volume used for point cloud binning and processing
//TODO: move this to json params
#define SUBCUBE_SIZE_M 0.01f
#define SUBCUBE_SIZE_MM 10
#define X_MIN (-2000/SUBCUBE_SIZE_MM)
#define X_MAX (2000/SUBCUBE_SIZE_MM)
#define Y_MIN (-2000/SUBCUBE_SIZE_MM)
#define Y_MAX (2000/SUBCUBE_SIZE_MM)
#define Z_MIN (-1000/SUBCUBE_SIZE_MM)
#define Z_MAX (2000/SUBCUBE_SIZE_MM)  
unsigned char binnedPointCloud[ (X_MAX-X_MIN)*(Y_MAX-Y_MIN)*(Z_MAX-Z_MIN)] = {0};

#define RANSAC_MAX_ITER  100
#define RANSAC_THRESHOLD  0.05f
#define RANSAC_DIVIDER 10

/*
  This is condensed reimlementation with some mods of ROS stereo_img_proc node functionality with images rectification.  

  HOWTO:

  To watch pointcloud published from this node:

  0. Enable publishToROS in RhIO
  1. On PC export ROS_MASTER_URI=http://192.168.xx.xx:11311 and launch rviz to view pointcloud

  If need to use ROS stereo processing instead of this one:
  0. Remove this filter from pipeline
  1. Make sure spinnaker source is configured to publish images and camerainfo
  2. Don't forget to export ROS_IP=192.168.xx.xx on robot before each ROS command to make ROS data aviable for remote PC for debug
  3. On robot run: ROS_NAMESPACE=camera rosrun stereo_image_proc stereo_image_proc _approximate_sync:=True
  4. On robot run: rosrun tf static_transform_publisher 0 0 0 0 0 -1 world cam_left_optical_frame 10 to view resultion pointcloud 
  5. On PC: export ROS_MASTER_URI=http://192.168.xx.xx:11311 and launch rviz to view pointcloud

  Calibration of the stereo pair using ROS (old):
  0. Run robot
  1. on PC run: ROS_MASTER_URI=http://192.168.xx.xx:11311
  2. On PC run:
   rosrun camera_calibration cameracalibrator.py --no-service-check --k-coefficients 6 --approximate 0.1 --size 9x7 --square 0.02 right:=/camera/right/image_raw left:=/camera/left/image_raw right_camera:=/camera/right left_camera:=/camera/left
  3.Save result on robot to /env/robot_name/calibration_stereo.json
  TODO: implement service /camera/left/set_camera_info to save calibration params via single "COMMIT" button click at ROS gui 

  Calibration of the stereo pair using python calibrator (new):
  0. Run robot
  1. On robot set doCalibration=true here in RhIO
  2. Resulting images will  be in /tmp
  3. Use them for /Vision/python/camera_calibration_stereo_chess.py
*/

/*
JSON usage:
        "filters" : [
            {
                "class name" : "SourceSpinnaker",
                "content" : {
                    "name" : "sourceRaw",
                    "warningExecutionTime": 0.05,
                    "shutter" : 5.0,
                    "gain" : 20.0,
                    "framerate" : 30.0,
		            "whitebalance_red" : 1.4,
		            "whitebalance_blue" : 3.0,
                    "camera_serial" : "19176738",
                    "is_primary" : true


                }
            },
            {
                "class name" : "SourceSpinnaker",
                "content" : {
                    "name" : "sourceRaw2",
                    "warningExecutionTime": 0.05,
                    "shutter" : 5.0,
                    "gain" : 20.0,
                    "framerate" : 30.0,
		            "whitebalance_red" : 1.4,
		            "whitebalance_blue" : 3.0,
                    "camera_serial" : "19176743",
                    "is_primary" : false

                }
            },
            {
                "class name" : "StereoImgProc",
                "content" : {
                    "name" : "stereoImgProc",
                    "dependencies" : ["sourceRaw", "sourceRaw2", "YBirdview"]
                }
            },
*/

namespace Vision
{
namespace Filters
{

class Parallel_disparity2pointcloud : public cv::ParallelLoopBody
{

private:
  cv::Mat_<cv::Vec3f> &dense_points_;
  cv::Mat &floatDisp;
  Matx44d &_Q;
  Eigen::Affine3d &cameraToSelf;


public:

    Parallel_disparity2pointcloud(cv::Mat_<cv::Vec3f> &dense_points_Val, cv::Mat &floatDispVal, Matx44d &_QVal, Eigen::Affine3d &cameraToSelfVal)
    : dense_points_(dense_points_Val), floatDisp(floatDispVal), _Q(_QVal), cameraToSelf(cameraToSelfVal)
    {
    }

    virtual void operator()(const cv::Range& range) const
    {

        
        int x, cols = floatDisp.cols;
        std::vector<float> _sbuf(cols);
        std::vector<Vec3f> _dbuf(cols);
        float* sbuf = &_sbuf[0];
        Vec3f* dbuf = &_dbuf[0];

        for(int y = range.start; y < range.end; y++)
        {
            float* sptr = sbuf;
            Vec3f* dptr = dbuf;
            sptr = floatDisp.ptr<float>(y);
            dptr = dense_points_.ptr<Vec3f>(y);
            
            for( x = 0; x < cols; x++)
            {
                //double d = sptr[x];
                double d = sptr[x] + 1.0; //For some reason opencv 3.2.0 StereoBM marks absent pixels with disp=-16 in CV_16S case, which equals -1 in float, but should be zero
                if(d!=0) {
                    Vec4d homg_pt = _Q*Vec4d(x, y, d, 1.0);
                    dptr[x] = Vec3d(homg_pt.val);
                    dptr[x] /= homg_pt[3]; //Delault reprojectImageTo3D gives SIGFPE here when d==0
                    Eigen::Vector3d posInCamera(dptr[x][0], dptr[x][1], dptr[x][2]);
                    Eigen::Vector3d posInSelf = cameraToSelf * posInCamera;
                    //Eigen::Vector3d posInSelf = posInCamera;
                    dptr[x][0] = posInSelf(0);
                    dptr[x][1] = posInSelf(1);
                    dptr[x][2] = posInSelf(2); 
                    //if(fabs(posInSelf(2) < 0.2)) plane_candidates.push_back(posInSelf);
                } else {
                    dptr[x][0] = 0;
                    dptr[x][1] = 0;
                    dptr[x][2] = MISSING_Z;
                }
            }
        }                    
    }
};


StereoImgProc::StereoImgProc() : Filter("StereoImgProc") 
{

  std::string path = "calibration_stereo.json"; 

  Json::Value json_content = rhoban_utils::file2Json(path);

  std::vector<double> Mlvec, Dlvec, Mrvec, Drvec;
  std::vector<double> Rlvec, Plvec, Rrvec, Prvec;

  rhoban_utils::tryReadVector(json_content, "Kl", &Mlvec);  Ml = cv::Mat(3, 3, CV_64F, Mlvec.data());
  rhoban_utils::tryReadVector(json_content, "Kr", &Mrvec);  Mr = cv::Mat(3, 3, CV_64F, Mrvec.data());

  rhoban_utils::tryReadVector(json_content, "Dl", &Dlvec);  Dl = cv::Mat(Dlvec);
  rhoban_utils::tryReadVector(json_content, "Dr", &Drvec);  Dr = cv::Mat(Drvec);

  //std::cout << "Dl=" << Dl << std::endl;
  //std::cout << "Dr=" << Dr << std::endl;

  cv::Size img_size(720, 540);

  #if USE_RECTIFICATION_FROM_JSON
  rhoban_utils::tryReadVector(json_content, "Rl", &Rlvec);  Rl = cv::Mat(Rlvec).reshape(3,3);
  rhoban_utils::tryReadVector(json_content, "Rr", &Rrvec);  Rr = cv::Mat(Rrvec).reshape(3,3);

  rhoban_utils::tryReadVector(json_content, "Pl", &Plvec);  Pl = cv::Mat(Plvec).reshape(4,3);
  rhoban_utils::tryReadVector(json_content, "Pr", &Prvec);  Pr = cv::Mat(Prvec).reshape(4,3);
  
  //Computing Q matrix from P same as the ROS does at http://docs.ros.org/api/image_geometry/html/c++/stereo__camera__model_8cpp_source.html
  double left_fx = Pl.at<double>(0,0);
  double left_fy = Pl.at<double>(1,1);
  double left_cx = Pl.at<double>(0,2);
  double left_cy = Pl.at<double>(1,2);
  double left_Tx = Pl.at<double>(0,3);
  double left_Ty = Pl.at<double>(1,3);

  double right_fx = Pr.at<double>(0,0);
  double right_fy = Pr.at<double>(1,1);
  double right_cx = Pr.at<double>(0,2);
  double right_cy = Pr.at<double>(1,2);
  double right_Tx = Pr.at<double>(0,3);
  double right_Ty = Pr.at<double>(1,3);
    
  double baseline = -right_Tx / right_fx;
  
  Q = cv::Mat (4,4, CV_32F);
  Q = 0;

  double Tx = -baseline; // The baseline member negates our Tx. Undo this negation
  Q.at<float>(0,0) =  left_fy * Tx;
  Q.at<float>(0,3) = -left_fy * left_cx * Tx;
  Q.at<float>(1,1) =  left_fx * Tx;
  Q.at<float>(1,3) = -left_fx * left_cy * Tx;
  Q.at<float>(2,3) =  left_fx * left_fy * Tx;
  Q.at<float>(3,2) = -left_fy;
  Q.at<float>(3,3) =  left_fy * (left_cx - right_cx); // zero when disparities are pre-adjusted  


  #else 
  //Read R and T from ROS json (ROS stereo_imag_proc doesnt use them, but calibrate_camera computes them)
  //Then calculate R,P,Q matrices here using custom alpha (fisheye scale) coeff for better FOV 
  //(ROS uses opencv automatic alpha calculation which is not optimal for rhoban fisheye lenses)
  std::vector<double> Rvec, Tvec;
  cv::Mat R, T;
  rhoban_utils::tryReadVector(json_content, "R", &Rvec);  R = cv::Mat(3, 3, CV_64F, Rvec.data());//.reshape(3);
  rhoban_utils::tryReadVector(json_content, "T", &Tvec);  T = cv::Mat(Tvec); //.reshape(3,1); 

  std::cout << "R=" << R << std::endl;
  std::cout << "T=" << T << std::endl;
  std::cout << "R.cols=" << R.cols << std::endl;
  std::cout << "R.rows=" << R.rows << std::endl;

  float alpha = -1.0;
  int flags = 0;  
  //flags |= CALIB_ZERO_DISPARITY;
  cv::stereoRectify(Ml, Dl, Mr, Dr, img_size, R, T, Rl, Rr, Pl, Pr, Q, flags, alpha, img_size);
  //cv::stereoRectify(M2, D2, M1, D1, img_size, R, T, R2, R1, P2, P1, Q, flags, alpha, img_size);
  //cv::stereoRectify(M2, D2, M1, D1, img_size, R, T, R2, R1, P2, P1, Q, flags, alpha, img_size);
  std::cout << "Q=" << Q << std::endl;  
  std::cout << "Rl=" << Rl << std::endl;
  std::cout << "Rr=" << Rr << std::endl;
  std::cout << "Pl=" << Pl << std::endl;
  std::cout << "Pr=" << Pr << std::endl;
  
  #endif

  std::cout << "Doing initUndistortRectifyMap..." << std::endl;
  cv::initUndistortRectifyMap(Ml, Dl, Rl, Pl, img_size, CV_16SC2, mapl1, mapl2);
  cv::initUndistortRectifyMap(Mr, Dr, Rr, Pr, img_size, CV_16SC2, mapr1, mapr2); 
  std::cout << "Done" << std::endl;

  //Below is to check the calibration of each camera without stereo rectification
  //cv::initUndistortRectifyMap(Ml, Dl, cv::Mat(), M1, img_size, CV_16SC2, mapl1, mapl2);
  //cv::initUndistortRectifyMap(Mr, Dr, cv::Mat(), Mr, img_size, CV_16SC2, mapr1, mapr2); 
  
  numberOfDisparities = ((img_size.width/8) + 15) & -16;

  int SADWindowSize = 15;

  #if USE_SGBM
  std::cout << "Creating StereoSGBM " << std::endl;
  sgbm = StereoSGBM::create(0,16,3);
  sgbm->setPreFilterCap(63);
  int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
  sgbm->setBlockSize(sgbmWinSize);
  int cn = 1;

  sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
  sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
  sgbm->setMinDisparity(0);
  sgbm->setNumDisparities(numberOfDisparities);
  sgbm->setUniquenessRatio(10);
  sgbm->setSpeckleWindowSize(100);
  sgbm->setSpeckleRange(32);
  sgbm->setDisp12MaxDiff(1);
  //sgbm->setMode(StereoSGBM::MODE_HH);
  sgbm->setMode(StereoSGBM::MODE_SGBM);
  //sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);
  #else
  
  std::cout << "Creating StereoBM " << std::endl;
  bm = StereoBM::create(16,9);
  bm->setPreFilterSize(9);
  bm->setPreFilterCap(31);
  bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
  bm->setMinDisparity(0);
  bm->setNumDisparities(numberOfDisparities);
  bm->setTextureThreshold(10);
  bm->setUniquenessRatio(15);
  bm->setSpeckleWindowSize(100);
  bm->setSpeckleRange(4);
  bm->setDisp12MaxDiff(1);  
  #endif
  std::cout << "Done!" << std::endl;
  
  std::cout << "Advertising /camera/spinnaker_cloud..." << std::endl;
  
  ros::NodeHandle nh;
  
  poincloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/camera/spinnaker_cloud", 1);  
  obstacles_bboxes_pub = nh.advertise<visualization_msgs::Marker>("/camera/obstacles", 1);
  camera_fov_pub = nh.advertise<visualization_msgs::Marker>("/camera/fov", 1);  
  plane_pub = nh.advertise<visualization_msgs::Marker>("/camera/ransac_plane", 1);

  std::cout << "Done!" << std::endl;

  obstaclesAndFreeSpacesHeightMap = cv::Mat(X_MAX-X_MIN, Y_MAX-Y_MIN,CV_8UC3);

}    


std::string StereoImgProc::getClassName() const
{
  return "StereoImgProc";
}

int StereoImgProc::expectedDependencies() const
{
  return 3;
}

void StereoImgProc::setParameters()
{
  //nbCols = ParamInt(4, 2, 200);
  //nbRows = ParamInt(4, 2, 200);
  //params()->define<ParamInt>("nbCols", &nbCols);
  //params()->define<ParamInt>("nbRows", &nbRows);
}

inline bool isValidPoint(const cv::Vec3f& pt)
{
  // Check both for disparities explicitly marked as invalid (where OpenCV maps pt.z to MISSING_Z)
  // and zero disparities (point mapped to infinity).
  return pt[2] != MISSING_Z; // && !std::isinf(pt[2]);
}

void StereoImgProc::process()
{
  bindProperties();

  importPropertiesFromRhIO();

  std::string rightName = _dependencies[0]; 
  cv::Mat rightRaw = (getDependency(rightName).getImg())->clone();

  std::string leftName = _dependencies[1]; 
  cv::Mat leftRaw = (getDependency(leftName).getImg())->clone();

  std::string birdviewName = _dependencies[2]; 
  cv::Mat birdview = (getDependency(birdviewName).getImg())->clone();

  //If spinnaker is not inited it will retun emty image of 640x480 and 1440x1080 when all is ok
  if(leftRaw.cols==640) return; //TODO remove this hardcode
  Benchmark::open("Downscale and cvtcolor input images");
  cv::Mat left;
  cv::Mat right;
  cv::Mat leftBGR;
  cv::Mat rightBGR;
  // Downsizing images if needed on live robot (they are 1440x1080), we can'd to stereo matching/chessboard detection fast enough on this image size 
  // TODO: make this use JSON or move pyrdown to separate pipeline filter    
  if(leftRaw.cols > 720) {
    cv::pyrDown(leftRaw, leftBGR);
  } else {
    leftRaw.copyTo(leftBGR);
  }
  if(rightRaw.cols > 720) {
    cv::pyrDown(rightRaw, rightBGR);
  } else {
    rightRaw.copyTo(rightBGR);  
  }


  cv::cvtColor(leftBGR, left, cv::COLOR_BGR2GRAY);
  cv::cvtColor(rightBGR, right, cv::COLOR_BGR2GRAY);
  Benchmark::close("Downscale and cvtcolor input images");

  Benchmark::open("Remap");
  cv::Mat left_r, right_r;
  remap(left, left_r, mapl1, mapl2, cv::INTER_LINEAR);
  remap(right, right_r, mapr1, mapr2, cv::INTER_LINEAR);

  Benchmark::close("Remap");
  int row_nb = left.rows;
  int col_nb = left.cols;

  Vision::Utils::CameraState *cs = &getCS();

  Eigen::Affine3d cameraToSelf = (cs->worldToSelf * cs->cameraToWorld);




  //Eigen::Matrix4d m = cs->worldToSelf.matrix();
  //std::cout << m << std::endl;

  if(debugDisplayRectification) {
    //Displays both rectified input images side-by-side and exits. Useful to check calibration quality
    //This is the exact imagages used for disparity calculation
    cv::Mat im(row_nb, col_nb*2, CV_8UC1, cv::Scalar(0,0,0));
    cv::Mat im_left_roi  = im(cv::Rect(     0, 0, col_nb, row_nb));
    cv::Mat im_right_roi = im(cv::Rect(col_nb, 0, col_nb, row_nb));

    left_r.copyTo(im_left_roi);
    right_r.copyTo(im_right_roi);
    if(debugPauseOn==false) { //Freeze lcurrentive input image for stop-frame analysis on live robot
      img() = im;
      im.copyTo(oldImg);
    } else {
      img() = oldImg;
    }
  } else 
  if(doCalibration) {
    cv::Mat im(row_nb, col_nb*2, CV_8UC3, cv::Scalar(0,0,0));
    
    cv::Mat im_left_roi  = im(cv::Rect(     0, 0, col_nb, row_nb));
    cv::Mat im_right_roi = im(cv::Rect(col_nb, 0, col_nb, row_nb));

    std::vector<Point2f> corners_left, corners_right;
    Size patternsize(pattern_size_w,pattern_size_h);
    bool patternfound_left = findChessboardCorners(left, patternsize, corners_left,
                          CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
                          + CALIB_CB_FAST_CHECK);
    bool patternfound_right = findChessboardCorners(right, patternsize, corners_right,
                          CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
                          + CALIB_CB_FAST_CHECK);
    if(patternfound_left && patternfound_right) {
      drawChessboardCorners(leftBGR, patternsize, Mat(corners_left), patternfound_left);
      drawChessboardCorners(rightBGR, patternsize, Mat(corners_right), patternfound_right);

      int fps_divider = 10;
      if(calibrationImgN % fps_divider == 0) {  
        char s[255];
        sprintf(s, "/tmp/left-%03d.png", calibrationImgN / fps_divider);
        cv::imwrite(s, left);
        sprintf(s, "/tmp/right-%03d.png", calibrationImgN / fps_divider); 
        cv::imwrite(s, right);
      }
      char s[255];
      sprintf(s, "saved frames:%d", calibrationImgN / fps_divider);
      int fontFace = cv::FONT_HERSHEY_PLAIN;
      double fontScale = 1.1;
      int thickness = 1;    
      cv::putText(leftBGR, std::string(s), cv::Point(0, 20), fontFace, fontScale, cv::Scalar::all(255), thickness, CV_AA);


      calibrationImgN++;
    }
    leftBGR.copyTo(im_left_roi);
    rightBGR.copyTo(im_right_roi); 
    img() = im;
 } else {
    Benchmark::open("StereoBM/SGBM");
    cv::Mat_<int16_t> disp;    
#if USE_SGBM
    sgbm->compute(left_r, right_r, disp);
#else
    bm->compute(left_r, right_r, disp);
#endif
    Benchmark::close("StereoBM/SGBM");

    if(debugShowDisparity) {
      Benchmark::open("debugShowDisparity");      
      Mat disp8;
      disp.convertTo(disp8, CV_8U, 4.0*255/(numberOfDisparities*16.));
      Mat dispJet;
      applyColorMap(disp8, dispJet, COLORMAP_JET);
      //img() = dispJet;
      img() = disp8;
      Benchmark::close("debugShowDisparity");
      return;
    }

    Benchmark::open("Preparing verbose birdview");
    //Verbose image size, can be arbitrary
    int width = birdview.cols;
    int height = birdview.rows;
    float verbosePixInOneMeter = 200.0; //200.0;
    
    //Rotate and scale birdview image to make it consistent with verbose scale 
    rhoban_utils::Angle yaw = cs->getYaw();
    double scale = verbosePixInOneMeter / (cs->getBirdviewPixelsInOneMeter()); //Magic number taken from ApproachPotential.h->pixInMeter;
    cv::Mat affine = cv::getRotationMatrix2D(cv::Point2f(width/2, height), yaw.getSignedValue(), scale);
    warpAffine(birdview, birdview, affine, birdview.size());
    cv::Mat birdviewColor;
    cv::cvtColor(birdview, birdviewColor, cv::COLOR_GRAY2BGR);
    Benchmark::close("Preparing verbose birdview");

    //img() = birdview;
    
    cv::Mat_<cv::Vec3f> dense_points_(disp.rows, disp.cols); 
    cv::Mat dense_points_indexes_in_cloud(disp.rows, disp.cols, CV_32SC1);
    dense_points_indexes_in_cloud.setTo(-1);

    
    Benchmark::open("Disparity to float");
    Mat floatDisp;
    float disparity_multiplier = 1.0f;
    if (disp.type() == CV_16S) disparity_multiplier = 16.0f; //Our case

    disp.convertTo(floatDisp, CV_32F, 1.0f / disparity_multiplier);
    Benchmark::close("Disparity to float");
    
    //cv::medianBlur(floatDisp, floatDisp, 3); //TODO: if this really helps?

    //Below is a custom parallelized implentation of cv::reprojectImageTo3D with hardcoded protection against zero disparity values.
    //Also pointcloud will be transformed from camera to self frame,
    //thus making field surface laying on z=0 plane for convinent obstacle detection, etc
    Benchmark::open("Disparity to pointcloud");
    Matx44d _Q;
    Q.convertTo(_Q, CV_64F);
    Parallel_disparity2pointcloud parallel_disparity2pointcloud(dense_points_, floatDisp, _Q, cameraToSelf);
    parallel_for_(Range(0, floatDisp.rows), parallel_disparity2pointcloud, PARALLEL_THREADS);
    Benchmark::close("Disparity to pointcloud");

    //points after translation: X looks forward, Y looks left, Z looks up - same as in ROS rviz

    //Binning point cloud
    Benchmark::open("Zeroing voxel volume");
    //TODO: use "zeroing prevoiusly modified values" instead of memset for speedup
    memset(binnedPointCloud, 0, (X_MAX-X_MIN)*(Y_MAX-Y_MIN)*(Z_MAX-Z_MIN)*sizeof(unsigned char));
    Benchmark::close("Zeroing voxel volume");

    Benchmark::open("Binning point cloud using voxel volume");
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    int32_t index = 0;
    for( int y = 0; y < dense_points_.rows; y++ )
    {
      for( int x = 0; x < dense_points_.cols; x++ )
      {
        cv::Vec3f point = dense_points_.at<cv::Vec3f>(y,x);
        if(point[2] != MISSING_Z) {
          int ix = (float)point[0] / SUBCUBE_SIZE_M;
          int iy = (float)point[1] / SUBCUBE_SIZE_M;
          int iz = (float)point[2] / SUBCUBE_SIZE_M;

          if( (ix >= X_MIN)&&(ix < X_MAX)&&(iy >= Y_MIN)&&(iy < Y_MAX)&&(iz >= Z_MIN)&&(iz < Z_MAX) ) {
            unsigned char *bin = &binnedPointCloud[(ix-X_MIN) + (iy-Y_MIN)*(X_MAX-X_MIN) + (iz-Z_MIN)*(X_MAX-X_MIN)*(Y_MAX-Y_MIN)];
            if(*bin==0) {
              //This is a empty volex, so let's add this point to the pointcloud
              *bin = 1;
              pcl::PointXYZRGB pt;
              pt.x = point[0];
              pt.y = point[1];
              pt.z = point[2] - 0.05; //Temporary lower points on verbose pointcloud published to ROS to make teb_local_planner trajectory more visible
              pt.r = left_r.at<uint8_t>(y,x);
              pt.g = left_r.at<uint8_t>(y,x);
              pt.b = left_r.at<uint8_t>(y,x);
              cloud->push_back(pt);
              dense_points_indexes_in_cloud.at<int32_t>(y,x) = index;
              index++;
              if(index >= 0x7FFFFFFF) throw std::logic_error("Unrealistic number of points in pointcloud");
            } else {
              //This voxel is already occupied, so skip this point
              //Mark this point as missing in dense_points array, making dense_points_ use constant spatial resolution
              dense_points_.at<cv::Vec3f>(y,x) =  cv::Vec3f(0,0,MISSING_Z); 
            }


          }  
        }
      }
    } 
    Benchmark::close("Binning point cloud using voxel volume");   

    Benchmark::open("RANSAC plane");
    //Filling plane candidates for RANSAC
    std::vector<cv::Vec3f> plane_candidates;
    for( int y = 0; y < dense_points_.rows; y++ )
    {
      for( int x = 0; x < dense_points_.cols; x++ )
      {
        cv::Vec3f point = dense_points_.at<cv::Vec3f>(y,x);
        if(point[2] != MISSING_Z) {
          float d_from_robot = sqrt(pow(point[0], 2) + pow(point[1], 2));
          if(d_from_robot > maxObstacleDistFromRobot) continue;  // Point too far from the robot, ignore it.
          //if((point[0] < maxObstacleDistFromRobot) /*&& (fabs(point[2] < minObstacleHeight))*/) 
          plane_candidates.push_back(point); 
        }
      }
    }
    cv::Vec3f normal;
    cv::Vec3f origin;
    cv::Vec3f best_plane_centroid;
    fitRansacPlane(plane_candidates, &normal, &origin, &best_plane_centroid);
    if(normal[2]<0) normal = -normal; //Make sure plane normal is facing up
    Benchmark::close("RANSAC plane");

    //Publishing fitted plane
    if(1) {

      //Converting plane axis to quaternion as in https://answers.ros.org/question/31006/how-can-a-vector3-axis-be-used-to-produce-a-quaternion/
      tf::Vector3 axis_vector(normal[0], normal[1], normal[2]);
      tf::Vector3 up_vector(0.0, 0.0, 1.0);
      tf::Vector3 right_vector = axis_vector.cross(up_vector);
      if(right_vector.length() > 0) {
        right_vector.normalized();
        tf::Quaternion q(right_vector, -1.0*acos(axis_vector.dot(up_vector)));
        q.normalize();
        geometry_msgs::Quaternion plane_orientation;
        tf::quaternionTFToMsg(q, plane_orientation);      

        visualization_msgs::Marker marker;
        marker.ns = "plane";
        marker.id = 0;
        marker.header.frame_id = "self";
        marker.type = visualization_msgs::Marker::CUBE;
      
        marker.pose.position.x = best_plane_centroid[0];
        marker.pose.position.y = best_plane_centroid[1];
        marker.pose.position.z = best_plane_centroid[2] - 0.05; //Temporary lower plane on verbose pointcloud published to ROS to make teb_local_planner trajectory more visible
        marker.pose.orientation = plane_orientation;

        marker.scale.x = maxObstacleDistFromRobot*2;
        marker.scale.y = maxObstacleDistFromRobot*2;
        marker.scale.z = RANSAC_THRESHOLD*2;

        marker.color.g = 1;
        marker.color.a = 0.1;
        plane_pub.publish(marker);
      }
    }

    obstaclesAndFreeSpacesHeightMap = 0;

    Benchmark::open("Filling cloud_above_ground");
    #if USE_FULL_3D_CLUSTERING
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_above_ground (new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud_above_ground->width = dense_points_.cols;
    cloud_above_ground->height = dense_points_.rows;
    cloud_above_ground->is_dense = false;  
    cloud_above_ground->points.resize (cloud_above_ground->width * cloud_above_ground->height);
    const float bad_point = std::numeric_limits<float>::quiet_NaN();
    pcl::PointIndices::Ptr cloud_above_ground_indices(new pcl::PointIndices());
    #endif

    int point_n = 0;
    for (int32_t u = 0; u < dense_points_.rows; ++u) {
        for (int32_t v = 0; v < dense_points_.cols; ++v) {
          bool obstacle = false;
          cv::Vec3f point = dense_points_.at<Vec3f>(u,v);
          if(point[2] != MISSING_Z) {
            cv::Vec3f M(point - origin);
            float d_from_plane = M.dot(normal); 
            float d_from_robot = sqrt(pow(point[0], 2) + pow(point[1], 2));
            if(d_from_plane > minObstacleHeight) obstacle = true; //Point above ground plane with enough threshold
            
            if(fabs(d_from_plane) > maxObstacleHeight) continue; // Ignoring obstacles with height more than 1.0 meter (usually it's a noise)
            if(fabs(d_from_robot) < minObstacleDistFromRobot) continue;
            if(fabs(d_from_robot) > maxObstacleDistFromRobot) continue;  // Point too far from the robot, ignore it.
          
            //Filling 2D obstacle maps
            int ix = (float)point[0] / SUBCUBE_SIZE_M;
            int iy = (float)point[1] / SUBCUBE_SIZE_M;
            if( (ix >= X_MIN)&&(ix < X_MAX)&&(iy >= Y_MIN)&&(iy < Y_MAX)) {
              unsigned char channel_n;
              if(obstacle) channel_n = 2; // Put obstacles in red channel              
              else channel_n = 1;         // Put free spaces in green channel
              int old_height = obstaclesAndFreeSpacesHeightMap.at<cv::Vec3b>(X_MAX-ix, Y_MAX-iy)[channel_n];
              // If there are several 3D points in this 2D map position - use lowest point height
              // because low obstacles are most dangerous (i.e. if this is a hand of an arbiter - we need to know there is it's lowest point)
              // Also this helps to filter "flying blobs" type of obstacles later in obstacleMap
              // Also use output height bias of 128 for better image visibility. Vertical resolution will be 1cm per one pixel intensity change              
              int new_height = std::min(255, 128 + (int)(d_from_plane * 100.0));
              int final_height;
              if(old_height==0) final_height = new_height;
              else final_height = std::min(old_height, new_height);
              obstaclesAndFreeSpacesHeightMap.at<cv::Vec3b>(X_MAX-ix, Y_MAX-iy)[channel_n] = final_height;
              
              //marking this point in pointcloud with red or green color
              int32_t index = dense_points_indexes_in_cloud.at<int32_t>(u,v);
              if (index >=0 ) { //Point can be in dense_points but not in the cloud if it's outside of binning volume. It will be marked as index=-1
                if(obstacle) 
                  (*cloud)[index].r = std::min((*cloud)[index].r + 50, 255);
                else 
                  (*cloud)[index].g = std::min((*cloud)[index].g + 50, 255);
              }
            }
          }

          #if USE_FULL_3D_CLUSTERING
          //Filling pointcloud for 3D euclidean clustering
          if(obstacle) {
            cloud_above_ground->points[point_n].x = point[0];
						cloud_above_ground->points[point_n].y = point[1];
						cloud_above_ground->points[point_n].z = point[2]; 
            uint8_t c = left_r.at<uint8_t>(u,v);
            cloud_above_ground->points[point_n].r = c;
						cloud_above_ground->points[point_n].g = c;
						cloud_above_ground->points[point_n].b = c;
            cloud_above_ground_indices->indices.push_back(point_n);
          } else {
						cloud_above_ground->points[point_n].x = bad_point;
						cloud_above_ground->points[point_n].y = bad_point;
						cloud_above_ground->points[point_n].z = bad_point;             
          }
          #endif
          point_n++;          
        }
    }    
    Benchmark::close("Filling cloud_above_ground");


    // Benchmark::open("Drawing verbose topview");
    // for (int32_t u = 0; u < dense_points_.rows; ++u) {
    //     for (int32_t v = 0; v < dense_points_.cols; ++v) {
    //       bool obstacle = false;
    //       Vec3f point = dense_points_.at<Vec3f>(u,v);
    //       int x = birdviewColor.cols - (point[1] * verbosePixInOneMeter + birdviewColor.cols/2);
    //       int y = birdviewColor.rows - point[0]*verbosePixInOneMeter;
    //       if((x>0) && (y>0) && (x<birdview.cols-1) && (y<birdview.rows-1)) {
    //         Vec3b c = birdviewColor.at<Vec3b>(y,x);
    //         float z = point[2];
    //         //birdviewColor.at<Vec3b>(y,x) = Vec3b(c[0], c[1], z*255.0);
    //         //Vec3b cp = Vec3b(left_r.at<uint8_t>(u,v), left_r.at<uint8_t>(u,v), c[2]);
    //         //if(z>0.05) obstacle = true;
    //         cv::Vec3f M(point - origin);
    //         float d_from_plane = M.dot(normal); 
    //         if(d_from_plane>0.05) obstacle = true;

    //         Vec3b cp = Vec3b(0,255,0);
    //         if(obstacle) cp = Vec3b(0,0,255);
    //         birdviewColor.at<Vec3b>(y,x) = cp;
    //         if(obstacle) {
    //           birdviewColor.at<Vec3b>(y+1,x) = cp;
    //           birdviewColor.at<Vec3b>(y,x+1) = cp;
    //           birdviewColor.at<Vec3b>(y+1,x+1) = cp;
    //         }
    //       }
    //     }
    //  }    
    // Benchmark::close("Drawing verbose topview");


    img() = obstaclesAndFreeSpacesHeightMap;

    


    Benchmark::open("Publishing full pointcloud");
    sensor_msgs::PointCloud2 msg_out;
    pcl::toROSMsg(*cloud, msg_out);
    msg_out.header.frame_id = "self";
    poincloud_pub.publish(msg_out);  
    Benchmark::close("Publishing full pointcloud");

    publishCameraFOV();

    

    #if USE_FULL_3D_CLUSTERING
    Benchmark::open("Creating KdTree for cloud_above_ground");
    // Creating the KdTree object for the search method of the extraction       
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud (cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(obstacle3DClusteringClusterTolerance);
    ec.setMinClusterSize(obstacle3DClusteringMinClusterSize);
    ec.setMaxClusterSize(obstacle3DClusteringMaxClusterSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_above_ground);
    ec.setIndices(cloud_above_ground_indices);
    Benchmark::close("Creating KdTree for cloud_above_ground");

    
    Benchmark::open("Euclidean cluster extraction");
    ec.extract (cluster_indices);
    //std::cout << "cluster_indices.size()=" << cluster_indices.size() << std::endl;
    Benchmark::close("Euclidean cluster extraction");


    Benchmark::open("Publishing obstacle bboxes");
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(cloud_above_ground);
    extract.setNegative(false);

    detectedObstacles.clear();
    // Getting camera state with isSlowStereo=true to use cameraState with selfToWorld matrix at the moment of stereo frame capturing, not current classic pipline cameraState
    Pipeline* pipeline = getPipeline();
    Utils::CameraState* cs = pipeline->getCameraState(true); 
    if(cs != nullptr) {
      for (size_t i = 0; i < cluster_indices.size(); ++i) {
        // Publish a bounding box around it.
        pcl::PointIndices::Ptr indices(new pcl::PointIndices);
        *indices = cluster_indices[i];

        visualization_msgs::Marker object_marker;
        object_marker.ns = "obstacle_bboxes";
        object_marker.id = i;
        object_marker.header.frame_id = "self";
        object_marker.type = visualization_msgs::Marker::CUBE;
        object_marker.lifetime = ros::Duration(0.1);
      
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cloud_above_ground, *indices, min_pt, max_pt);

        object_marker.pose.position.x = (max_pt.x() + min_pt.x()) / 2;
        object_marker.pose.position.y = (max_pt.y() + min_pt.y()) / 2;
        object_marker.pose.position.z = (max_pt.z() + min_pt.z()) / 2;
        object_marker.pose.orientation.w = 1;

        object_marker.scale.x = max_pt.x() - min_pt.x();
        object_marker.scale.y = max_pt.y() - min_pt.y();
        object_marker.scale.z = max_pt.z() - min_pt.z();

        object_marker.color.r = 1;
        object_marker.color.a = 0.3;
        obstacles_bboxes_pub.publish(object_marker);

        //Also exporting 3D clustered obstacles to placer
        StereoCircularObstacle obstacle;
        Eigen::Vector3d pos_in_self((max_pt.x() + min_pt.x()) / 2, (max_pt.y() + min_pt.y()) / 2, 0);
        
        //Eigen::Vector3d pos_in_world = getCS().getWorldFromSelf(pos_in_self);
        
        Eigen::Vector3d pos_in_world = cs->getWorldFromSelf(pos_in_self);
        
        obstacle.pos_in_world.x() = pos_in_world.x();
        obstacle.pos_in_world.y() = pos_in_world.y();
        obstacle.radius = ( (max_pt.x() + min_pt.x()) / 2 + (max_pt.y() + min_pt.y()) / 2 ) / 2;
        detectedObstacles.push_back(obstacle);
      }
    } else {
      throw std::logic_error("cameraState(slowStereo=true) for stereoImgProc is NULL");
    }
    Benchmark::close("Publishing obstacle bboxes");
    #endif   

          

       
    #if 0
    if(publishToROS) {
      Benchmark::open("Filling ROS::PointCloud2");
      sensor_msgs::PointCloud2 points;
      // Fill in sparse point cloud message
      points.header.frame_id = "self";
      points.height = dense_points_.rows;
      points.width  = dense_points_.cols;
      points.fields.resize (4);
      points.fields[0].name = "x";
      points.fields[0].offset = 0;
      points.fields[0].count = 1;
      points.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
      points.fields[1].name = "y";
      points.fields[1].offset = 4;
      points.fields[1].count = 1;
      points.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
      points.fields[2].name = "z";
      points.fields[2].offset = 8;
      points.fields[2].count = 1;
      points.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
      points.fields[3].name = "rgb";
      points.fields[3].offset = 12;
      points.fields[3].count = 1;
      points.fields[3].datatype = sensor_msgs::PointField::FLOAT32;
      //points.is_bigendian = false; ???
      points.point_step = 16;
      points.row_step = points.point_step * points.width;
      points.data.resize (points.row_step * points.height);
      points.is_dense = false; // there may be invalid points
      
      float bad_point = std::numeric_limits<float>::quiet_NaN ();
      int i;
      i = 0;
      for (int32_t u = 0; u < dense_points_.rows; ++u) {
          for (int32_t v = 0; v < dense_points_.cols; ++v, ++i) {
          if (isValidPoint(dense_points_.at<Vec3f>(u,v))) {
              // x,y,z,rgba
              memcpy (&points.data[i * points.point_step + 0], &dense_points_.at<Vec3f>(u,v)[0], sizeof (float));
              memcpy (&points.data[i * points.point_step + 4], &dense_points_.at<Vec3f>(u,v)[1], sizeof (float));
              memcpy (&points.data[i * points.point_step + 8], &dense_points_.at<Vec3f>(u,v)[2], sizeof (float));
          }
          else {
              //TODO: this seems doesn't work, all points are present in pointcloud 
              memcpy (&points.data[i * points.point_step + 0], &bad_point, sizeof (float));
              memcpy (&points.data[i * points.point_step + 4], &bad_point, sizeof (float));
              memcpy (&points.data[i * points.point_step + 8], &bad_point, sizeof (float));
          }
          }
      }
      // Fill in color
      i = 0;
      //RGB
      /*for (int32_t u = 0; u < dense_points_.rows; ++u) {
        for (int32_t v = 0; v < dense_points_.cols; ++v, ++i) {
          if (isValidPoint(dense_points_(u,v))) {
            const cv::Vec3b& bgr = left_r.at<cv::Vec3b>(u,v);
            int32_t rgb_packed = (bgr[2] << 16) | (bgr[1] << 8) | bgr[0];
            memcpy (&points.data[i * points.point_step + 12], &rgb_packed, sizeof (int32_t));
          }
          else {
            memcpy (&points.data[i * points.point_step + 12], &bad_point, sizeof (float));
          }
        }
      }*/
      //MONO
      for (int32_t u = 0; u < dense_points_.rows; ++u) {
        for (int32_t v = 0; v < dense_points_.cols; ++v, ++i) {
          if (isValidPoint(dense_points_(u,v))) {
            uint8_t g = left_r.at<uint8_t>(u,v);
            int32_t rgb = (g << 16) | (g << 8) | g;
            memcpy (&points.data[i * points.point_step + 12], &rgb, sizeof (int32_t));
          }
          else {
            memcpy (&points.data[i * points.point_step + 12], &bad_point, sizeof (float));
          }
        }
      }
      Benchmark::close("Filling ROS::PointCloud2");
      poincloud_pub.publish(points);
    }
    #endif
  }


}

void StereoImgProc::bindProperties()
{
  std::string filter_path = rhio_path + getName();
  if (RhIO::Root.getValueType(filter_path) != RhIO::ValueType::NoValue)
    return;
  RhIO::IONode& node = RhIO::Root.child(filter_path);
  //node.newFloat("whitebalance_red")->defaultValue(whitebalance_red); 
  node.newBool("publishToROS")->defaultValue(publishToROS);
  node.newBool("debugDisplayRectification")->defaultValue(debugDisplayRectification);
  node.newBool("doCalibration")->defaultValue(doCalibration);
  node.newBool("debugPauseOn")->defaultValue(debugPauseOn);
  node.newBool("debugShowDisparity")->defaultValue(debugShowDisparity);
  node.newFloat("minObstacleHeight")->defaultValue(minObstacleHeight);
  node.newFloat("maxObstacleHeight")->defaultValue(maxObstacleHeight);
  node.newFloat("minObstacleDistFromRobot")->defaultValue(minObstacleDistFromRobot);
  node.newFloat("maxObstacleDistFromRobot")->defaultValue(maxObstacleDistFromRobot); 
  node.newFloat("obstacle3DClusteringClusterTolerance")->defaultValue(obstacle3DClusteringClusterTolerance); 
  node.newFloat("obstacle3DClusteringMinClusterSize")->defaultValue(obstacle3DClusteringMinClusterSize); 
  node.newFloat("obstacle3DClusteringMaxClusterSize")->defaultValue(obstacle3DClusteringMaxClusterSize); 
}

void StereoImgProc::importPropertiesFromRhIO()
{
  std::string filter_path = rhio_path + getName();
  RhIO::IONode& node = RhIO::Root.child(filter_path);
  //whitebalance_red = node.getValueFloat("whitebalance_red").value;
  publishToROS = node.getValueBool("publishToROS").value;
  debugDisplayRectification = node.getValueBool("debugDisplayRectification").value;
  doCalibration = node.getValueBool("doCalibration").value;
  debugPauseOn = node.getValueBool("debugPauseOn").value;
  debugShowDisparity = node.getValueBool("debugShowDisparity").value;
  minObstacleHeight = node.getValueFloat("minObstacleHeight").value;
  maxObstacleHeight = node.getValueFloat("maxObstacleHeight").value;
  minObstacleDistFromRobot = node.getValueFloat("minObstacleDistFromRobot").value;
  maxObstacleDistFromRobot = node.getValueFloat("maxObstacleDistFromRobot").value;  
  obstacle3DClusteringClusterTolerance = node.getValueFloat("obstacle3DClusteringClusterTolerance").value;
  obstacle3DClusteringMinClusterSize = node.getValueFloat("obstacle3DClusteringMinClusterSize").value;
  obstacle3DClusteringMaxClusterSize = node.getValueFloat("obstacle3DClusteringMaxClusterSize").value;

}


void StereoImgProc::fitRansacPlane(const std::vector<cv::Vec3f> &points, cv::Vec3f *normal, cv::Vec3f *origin, cv::Vec3f *best_plane_centroid)
{


  size_t N = points.size();
  size_t best_inliers = 0;
  if(N<3) return;
  
  cv::Vec3f best_plane_normal;
  cv::Vec3f best_plane_p;

  for (size_t iterations = 0; iterations < RANSAC_MAX_ITER; iterations ++) {
    //1. Choose 3 points
    size_t i0, i1, i2;
    i0 = std::rand() % N;
    i1 = std::rand() % N;
    i2 = std::rand() % N;
    while(i1==i0) i1 = std::rand() % N;
    while((i2==i0)||(i2==i1)) i2 = std::rand() % N;

    //2. Make plane consensus
    cv::Vec3f v1 = points[i1] - points[i0];
    cv::Vec3f v2 = points[i2] - points[i0];
    
    cv::Vec3f n = v1.cross(v2);
    double l = cv::norm(n);
    if (l == 0.0) continue; //collinear points
    cv::Vec3f plane_normal(n/l);
    cv::Vec3f plane_p = points[i0]; 
    cv::Vec3f centroid(0,0,0);

    //3. Check consensus
    size_t inliers_count = 0;
    size_t divider = RANSAC_DIVIDER; //Cheach each N'th point only
    for (size_t i=0; i<N/divider; i++) {
      cv::Vec3f M(points[i*divider] - plane_p);
      float d = M.dot(plane_normal); 
      if (fabs(d) < RANSAC_THRESHOLD) {
        inliers_count++;
        centroid[0] += points[i*divider][0];
        centroid[1] += points[i*divider][1];
        centroid[2] += points[i*divider][2];
      }
    } 
    if(inliers_count > best_inliers) {
      best_inliers = inliers_count;
      best_plane_normal = plane_normal;
      best_plane_p = plane_p;
      (*best_plane_centroid)[0] = centroid[0]/inliers_count;
      (*best_plane_centroid)[1] = centroid[1]/inliers_count;
      (*best_plane_centroid)[2] = centroid[2]/inliers_count;
    }
  }

  //std::cout << "Plane RANSAC: inliers=" << best_inliers << ", normal=" << best_plane_normal << ", best_plane_p=" << best_plane_p << std::endl;
  *normal = best_plane_normal;
  *origin = best_plane_p;
}

//Publish camera central ray (for stereo setup this is the central ray of left cam)  
void StereoImgProc::publishCameraFOV()
{
  Vision::Utils::CameraState *cs = &getCS();

  visualization_msgs::Marker line_list;

  line_list.header.frame_id = "self";
  //line_list.ns = "my_namespace";
  line_list.id = 1;
  line_list.type = visualization_msgs::Marker::LINE_LIST;
  line_list.header.stamp = ros::Time::now();
  line_list.action = visualization_msgs::Marker::ADD;

  line_list.pose.orientation.w = 1.0;

  // LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
  line_list.scale.x = 0.01;

  // Line list is blue
  line_list.color.b = 1.0;
  line_list.color.a = 1.0;

  geometry_msgs::Point pc, p0, p1, p2, p3;

  Eigen::Vector3d cam_in_cam(0,0,0);
  Eigen::Vector3d cam_in_self = cs->worldToSelf * cs->cameraToWorld * cam_in_cam;
  
  pc.x = cam_in_self.x(); pc.y = cam_in_self.y(); pc.z = cam_in_self.z();

  Eigen::Vector3d center_ray_in_self = cs->worldToSelf.linear() * cs->cameraToWorld.linear() * Eigen::Vector3d::UnitZ();
  geometry_msgs::Point pr;
  double len = 2.0;
  pr.x = pc.x + len*center_ray_in_self.x();
  pr.y = pc.y + len*center_ray_in_self.y();
  pr.z = pc.z + len*center_ray_in_self.z();

  line_list.points.push_back(pc);
  line_list.points.push_back(pr);

  camera_fov_pub.publish(line_list);
  
}

Json::Value StereoImgProc::toJson() const
{
  Json::Value v = Filter::toJson();
  v["calibration_pattern_w"] = pattern_size_w;
  v["calibration_pattern_h"] = pattern_size_h;
  return v;
}

void StereoImgProc::fromJson(const Json::Value& v, const std::string& dir_name)
{
  Filter::fromJson(v, dir_name);
  rhoban_utils::tryRead(v, "calibration_pattern_w", &pattern_size_w);
  rhoban_utils::tryRead(v, "calibration_pattern_h", &pattern_size_h);
}

}  // namespace Filters
}  // namespace Vision

#include "Filters/Basics/Undistort.hpp"

#include "CameraState/CameraState.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "rhoban_utils/timing/benchmark.h"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>

#define UNDISTORT_ONLY 0

using rhoban_utils::Benchmark;

namespace Vision
{
namespace Filters
{
using namespace cv;
using namespace std;

class MyData
{
public:
  MyData()
  {
  }

  void write(FileStorage& fs) const  // Write serialization for this class
  {
    std::cout << "Write not implemented" << std::endl;
  }
  void read(const FileNode& node)  // Read serialization for this class
  {
    /*    A = (int)node["A"];
          X = (double)node["X"];
          id = (string)node["id"];*/
  }

public:
  Mat mat1;
  Mat mat2;
};

Undistort::Undistort() : Filter("Undistort")
{
  _first = true;
}

cv::Point Undistort::predecessor(const cv::Point& p) const
{
  cv::Point result;
  result.x = _map1Inverted.at<cv::Vec2s>(p)[0];
  result.y = _map1Inverted.at<cv::Vec2s>(p)[1];
  return result;
}

void Undistort::setParameters()
{
}

cv::Mat Undistort::undistortOneShot(cv::Mat input)
{
  cv::Mat output;
  cv::remap(input, output, _map1Inverted, _map2, cv::INTER_LINEAR);
  return output;
}

void Undistort::process()
{
  Benchmark::open("input");
  cv::Mat input_org = *(getDependency().getImg());
  cv::Mat input = input_org.clone();
  Benchmark::close("input");

  cv::Mat cameraMatrix;
  cv::Mat distCoeffs;
  cv::Size imageSize;
  
  //imageSize = input.size();   //Will fail bacause of on live robot first frame is empty
  imageSize = getCS().getCameraModel(Vision::Utils::CAMERA_WIDE_FULL).getImgSize();
  double alpha = 0.3;  

  //cv::pyrDown(input, img());
  //return;

  /*float arrayCameraMatrix[9] = { 581.3970227650393, 0.0,  751.7370642808319, 
                                 0.0, 582.8205343524182, 566.078183600938,
                                 0.0, 0.0, 1.0};
  
  cameraMatrix = cv::Mat(3, 3, CV_32F, arrayCameraMatrix);

  float arrayDistCoeffs[14] = {  0.27726319334217364, -0.05517851424216206, 7.05739711437462e-05, 6.859146803518088e-05,
                                -0.00179427848687631, 0.5969369412365509, -0.0412554412084885, -0.011590321634541451,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  
  distCoeffs= cv::Mat(14, 1, CV_32F, arrayDistCoeffs);    */

  cameraMatrix = getCS().getCameraModel(Vision::Utils::CAMERA_WIDE_FULL).getCameraMatrix();
  distCoeffs = getCS().getCameraModel(Vision::Utils::CAMERA_WIDE_FULL).getDistortionCoeffs();

  if (_first)
  {
    _first = false;
    Benchmark::open("initUndistortRectifyMap (should be done only once)");

    // calculating the correction matrix only once

    //std::cout << cameraMatrix << std::endl;
    //std::cout << distCoeffs << std::endl;

     cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),
                                 getOptimalNewCameraMatrix(cameraMatrix,
                                 distCoeffs, imageSize, alpha, imageSize, 0),
                                 imageSize, CV_16SC2, _map1, _map2);

    Benchmark::close("initUndistortRectifyMap (should be done only once)");

  }

  //Generate perspective transfom frustum from projected union square 1m x 1m

  Eigen::Vector3d flr;
  Eigen::Vector3d frr;  
  Eigen::Vector3d brr;  
  Eigen::Vector3d blr; 

  getCS().getBirdviewRotatedUnionSquareCorners(&flr, &frr, &brr, &blr);

  cv::Point2f self_pos_on_img_fl;
  cv::Point2f self_pos_on_img_fr;
  cv::Point2f self_pos_on_img_bl;
  cv::Point2f self_pos_on_img_br;

  try {
    self_pos_on_img_fl = getCS().imgXYFromWorldPosition(flr, Vision::Utils::CAMERA_WIDE_FULL);
    self_pos_on_img_fr = getCS().imgXYFromWorldPosition(frr, Vision::Utils::CAMERA_WIDE_FULL);
    self_pos_on_img_bl = getCS().imgXYFromWorldPosition(blr, Vision::Utils::CAMERA_WIDE_FULL);
    self_pos_on_img_br = getCS().imgXYFromWorldPosition(brr, Vision::Utils::CAMERA_WIDE_FULL);
    //circle(input, self_pos_on_img_fl, 10, Vec3b(0,0,255), -1);
    //circle(input, self_pos_on_img_fr, 10, Vec3b(255,0,255), -1);
    //circle(input, self_pos_on_img_bl, 10, Vec3b(255,0,255), -1);
    //circle(input, self_pos_on_img_br, 10, Vec3b(255,0,255), -1);
    //cv::line(input, self_pos_on_img_fl, self_pos_on_img_fr, Vec3b(255,0,255), 2); 
    //cv::line(input, self_pos_on_img_fr, self_pos_on_img_bl, Vec3b(255,0,255), 2);
    //cv::line(input, self_pos_on_img_bl, self_pos_on_img_br, Vec3b(255,0,255), 2);
    //cv::line(input, self_pos_on_img_br, self_pos_on_img_fl, Vec3b(255,0,255), 2);    
  } catch (const std::runtime_error& exc) {
    std::cout << "Cannot calculate birdview, union square is not in the image" << std::endl;
    img() = 255;
    return;
  }

  std::vector<Point2f> src_vertices_distorted;
  std::vector<Point2f> src_vertices_undistorted;

  //src_vertices[0] = Point2f(_map1.at<float>(self_pos_on_img_fl.y, self_pos_on_img_fl.x), _map2.at<float>(self_pos_on_img_fl.y, self_pos_on_img_fl.x));
  //src_vertices[1] = Point2f(_map1.at<float>(self_pos_on_img_fr.y, self_pos_on_img_fr.x), _map2.at<float>(self_pos_on_img_fr.y, self_pos_on_img_fr.x));
  //src_vertices[2] = Point2f(_map1.at<float>(self_pos_on_img_bl.y, self_pos_on_img_bl.x), _map2.at<float>(self_pos_on_img_bl.y, self_pos_on_img_bl.x));
  //src_vertices[3] = Point2f(_map1.at<float>(self_pos_on_img_br.y, self_pos_on_img_br.x), _map2.at<float>(self_pos_on_img_br.y, self_pos_on_img_br.x));
  src_vertices_distorted.push_back(self_pos_on_img_fl);
  src_vertices_distorted.push_back(self_pos_on_img_fr);
  src_vertices_distorted.push_back(self_pos_on_img_br);
  src_vertices_distorted.push_back(self_pos_on_img_bl);

  for (size_t i=0; i<src_vertices_distorted.size(); i++ ) {
    if( (src_vertices_distorted[i].x < 0)||(src_vertices_distorted[i].x >= imageSize.width) || 
        (src_vertices_distorted[i].y < 0)||(src_vertices_distorted[i].y >= imageSize.height) ) {
          std::cout << "Cannot calculate birdview, union square is not in the image" << std::endl;
          img() = 255; // dirty hack: return image filled with constant blue color
          return;          
        }
  }



  cv::undistortPoints(src_vertices_distorted, src_vertices_undistorted, cameraMatrix, distCoeffs, cv::Mat(), 
                        getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha, imageSize, 0));

  Point2f src_vertices[4];
  Point2f dst_vertices[4];
  src_vertices[0] = src_vertices_undistorted[0];
  src_vertices[1] = src_vertices_undistorted[1];
  src_vertices[2] = src_vertices_undistorted[2];
  src_vertices[3] = src_vertices_undistorted[3];

  //cv::Size outputSize = input.size()/2;  
  //cv::Point2f unionSquareCenterOnBirdview = cv::Point2f(outputSize.width/2, outputSize.height/2+100);
  //getCS().unionSquareCenterOnBirdview = unionSquareCenterOnBirdview;

  double birdviewPixelsInOneMeter = getCS().getBirdviewPixelsInOneMeter();
  cv::Point2f unionSquareCenterOnBirdviewPix = getCS().getUnionSquareCenterOnBirdviewPix();



  dst_vertices[0] = unionSquareCenterOnBirdviewPix + cv::Point2f(-birdviewPixelsInOneMeter/2, -birdviewPixelsInOneMeter/2); //fl
  dst_vertices[1] = unionSquareCenterOnBirdviewPix + cv::Point2f(birdviewPixelsInOneMeter/2, -birdviewPixelsInOneMeter/2); //fr
  dst_vertices[2] = unionSquareCenterOnBirdviewPix + cv::Point2f(birdviewPixelsInOneMeter/2, birdviewPixelsInOneMeter/2); //br
  dst_vertices[3] = unionSquareCenterOnBirdviewPix + cv::Point2f(-birdviewPixelsInOneMeter/2, birdviewPixelsInOneMeter/2); //bl

  /*cv::Point2f checkValue1 = getCS().robotPosFromBirdviewImg(unionSquareCenterOnBirdview.x-birdviewPixelsInOneMeter/2, unionSquareCenterOnBirdview.y-birdviewPixelsInOneMeter/2);
  cv::Point2f checkValue2 = getCS().robotPosFromBirdviewImg(unionSquareCenterOnBirdview.x+birdviewPixelsInOneMeter/2, unionSquareCenterOnBirdview.y-birdviewPixelsInOneMeter/2);
  cv::Point2f checkValue3 = getCS().robotPosFromBirdviewImg(unionSquareCenterOnBirdview.x+birdviewPixelsInOneMeter/2, unionSquareCenterOnBirdview.y+birdviewPixelsInOneMeter/2);
  cv::Point2f checkValue4 = getCS().robotPosFromBirdviewImg(unionSquareCenterOnBirdview.x-birdviewPixelsInOneMeter/2, unionSquareCenterOnBirdview.y+birdviewPixelsInOneMeter/2);
  cv::Point2f checkValue5 = getCS().robotPosFromBirdviewImg(unionSquareCenterOnBirdview.x-birdviewPixelsInOneMeter, unionSquareCenterOnBirdview.y-birdviewPixelsInOneMeter);

  std::cout << "robotPosFromBirdviewImg(p0)=" << checkValue1 << std::endl;
  std::cout << "robotPosFromBirdviewImg(p1)=" << checkValue2 << std::endl;
  std::cout << "robotPosFromBirdviewImg(p2)=" << checkValue3 << std::endl;
  std::cout << "robotPosFromBirdviewImg(p3)=" << checkValue4 << std::endl;
  std::cout << "robotPosFromBirdviewImg(p4)=" << checkValue5 << std::endl;*/

  cv::Mat M = getPerspectiveTransform(src_vertices, dst_vertices);


  /*cv::Point tl = cv::Point((float)source.cols*0.3, (float)source.rows*0.5);
  cv::Point tr = cv::Point((float)source.cols*0.7, (float)source.rows*0.5);
  cv::Point bl = cv::Point((float)source.cols*0.3, (float)source.rows*0.8);
  cv::Point br = cv::Point((float)source.cols*0.7, (float)source.rows*0.8);

  cv::line(img(), tl, tr, Vec3b(0,0,255), 2); 
  cv::line(img(), tr, bl, Vec3b(0,0,255), 2);
  cv::line(img(), bl, br, Vec3b(0,0,255), 2);
  cv::line(img(), br, tl, Vec3b(0,0,255), 2);
  */

  cv::Mat input_undistorted;
  if (Filter::GPU_ON)
  {
    Benchmark::open("OPENCL REMAP");
    cv::UMat image_gpu, image_gpu_undist;
    input.copyTo(image_gpu);

    cv::remap(image_gpu, image_gpu_undist, _map1, _map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    image_gpu_undist.copyTo(img());
    // std::cerr<<"OPENCL"<<std::endl;

    Benchmark::close("OPENCL REMAP");
  }
  else
  {
    Benchmark::open("remap");
    //input.copyTo(img());
    //cv::remap(input, img(), _map1, _map2, cv::INTER_LINEAR); 
    cv::remap(input, input_undistorted, _map1, _map2, cv::INTER_LINEAR);
    Benchmark::close("remap");
  }

  /*for(size_t i=0;i<4;i++) {
    cv::line(img(), src_vertices[i], src_vertices[(i+1)%4], Vec3b(0,0,255), 2); 
  }*/
  #if UNDISTORT_ONLY
    cv::pyrDown(input_undistorted, img());
  #else  
    warpPerspective(input_undistorted, img(), M, getCS().getBirdviewImageSize(), INTER_LINEAR, BORDER_CONSTANT);
  #endif
 
}
}  // namespace Filters
}  // namespace Vision

#include "Filters/Custom/ObstacleMap.hpp"
#include "CameraState/CameraState.hpp"
#include "robocup_referee/constants.h"
#include "rhoban_utils/timing/benchmark.h"

#define OBSTACLES_AS_ELLIPSES 0
#define OBSTACLES_AS_CONVEX_HULLS 1

//X looks forward, Y looks left, Z looks up
//This is a size and resolution of voxel volume used for obstacle processing
//TODO: move this to json params
#define SUBCUBE_SIZE_M 0.02f
#define SUBCUBE_SIZE_MM (1000.f*SUBCUBE_SIZE_M)
#define X_MIN (-2000/SUBCUBE_SIZE_MM)
#define X_MAX (2000/SUBCUBE_SIZE_MM)
#define Y_MIN (-2000/SUBCUBE_SIZE_MM)
#define Y_MAX (2000/SUBCUBE_SIZE_MM)
#define Z_MIN (-1000/SUBCUBE_SIZE_MM)
#define Z_MAX (2000/SUBCUBE_SIZE_MM)  

using namespace std;
using namespace cv;
using rhoban_utils::Benchmark;

namespace Vision
{
namespace Filters
{
ObstacleMap::ObstacleMap() : Filter(" ")
{
  accumulatedObstaclesWorld.clear();

}

ObstacleMap::~ObstacleMap()
{
}

void ObstacleMap::setParameters()
{
  /* define the size of the squared kernel side in pixel */
  /*kernel_size = ParamInt(30, 2, 100);
  params()->define<ParamInt>("kernel_size", &kernel_size);
  resize_factor = ParamFloat(1.0, 0.0, 1.0);
  params()->define<ParamFloat>("resize_factor", &resize_factor);*/
}

void ObstacleMap::process()
{
  
  std::string obstaclesAndFreeSpacesHeightMapName = _dependencies[0];
  cv::Mat obstaclesAndFreeSpacesHeightMap = (getDependency(obstaclesAndFreeSpacesHeightMapName).getImg())->clone();
  
  // Accumulating obstacles with 2cm resolution for proper processing speed, not 1cm as in stereoImgProc
  cv::pyrDown(obstaclesAndFreeSpacesHeightMap, obstaclesAndFreeSpacesHeightMap);

  if( (obstaclesAndFreeSpacesHeightMap.cols != X_MAX-X_MIN) && (obstaclesAndFreeSpacesHeightMap.rows != Y_MAX-Y_MIN) ) {
    // Input is not an obstacle height map (i.e. stereoImgProc debug verbose output is enabled)
    return;
  }

  Mat bgr[3];
  cv::split(obstaclesAndFreeSpacesHeightMap, bgr);
  cv::Mat obstaclesHeightMap(bgr[2]); // Extracting obstacles from red channel
  cv::Mat freeSpacesHeightMap(bgr[1]); // Extracting free spaces from green channel

  // StereoImgProc cam produce points which is both obstacle and free space sumultaniously (in case ov overhanging, etc)
  // So we need to clear accumulated obstacle points only if fresh point is marked as free space and not obstacle
  freeSpacesHeightMap.setTo(0, obstaclesHeightMap);  

  Vision::Utils::CameraState *cs = &getCS();
  Eigen::Affine3d worldToSelf = cs->worldToSelf;
  Eigen::Affine3d selfToWorld = worldToSelf.inverse();

  //0. Doing clustering and filtration of fresh stereo image (in form of top-view heightmap)
  std::vector<std::vector<cv::Point> > newFrameHulls;
  std::vector<cv::RotatedRect> newFrameMinRects;    
  clusterAndFilterObstaclesByConnectedComponents(obstaclesHeightMap, newFrameHulls, newFrameMinRects);

  // Drawing filtered fresh obstacles as filled convex hulls on current obstacleMap
  cv::Mat obstaclesNewFrame(X_MAX-X_MIN, Y_MAX-Y_MIN, CV_8UC1);  
  obstaclesNewFrame = 0;  
  for( int i = 0; i< newFrameMinRects.size(); i++ ) {
    cv::fillConvexPoly(obstaclesNewFrame, newFrameHulls[i], 255); // Drawing as obstacle
    cv::fillConvexPoly(freeSpacesHeightMap, newFrameHulls[i], 0); // Clearing free spaces on new frame according to convex hull
  }

  // 1. Transform new obstacles to world frame 
  std::vector<AccumulatedObstaclePoint> newObstaclesWithAgesWorld; 
  newObstaclesWithAgesWorld.clear();
  for(int col = 0; col < obstaclesNewFrame.cols; col++) {
    for(int row = 0; row < obstaclesNewFrame.rows; row++) {
      if(obstaclesNewFrame.at<unsigned char>(row, col) != 0) {
        //This is an obstacle        
        Eigen::Vector3d ps( (X_MAX - row)*SUBCUBE_SIZE_M,  (Y_MAX - col)*SUBCUBE_SIZE_M, 0);
        Eigen::Vector3d pw = selfToWorld * ps;
        AccumulatedObstaclePoint aop;
        aop.posWorld = pw;
        aop.age = 255; //New obstacle has an age of 255
        aop.row = row;
        aop.col = col;
        //newObstaclesWithAgesWorld.push_back(cv::Vec3d(pw.x(), pw.y(), 255)); //New obstacle has an age of 255
        newObstaclesWithAgesWorld.push_back(aop);
      }
    }
  }  

  //2. Decrease age of accumulated obstacles from previous frames, delete if age is negative
  std::vector<AccumulatedObstaclePoint>::iterator iter;
  for (iter = accumulatedObstaclesWorld.begin(); iter != accumulatedObstaclesWorld.end(); ) {
    double age = iter->age;

    // Decreasing age per frame
    age -= ageDeltaPerFrame;

    // Also decreasing age of accumulated obstacle point if it's marked as free space in fresh frame
    if(freeSpacesHeightMap.at<unsigned char>(iter->row, iter->col) != 0) 
      age -= ageDeltaPerFrame*2; 

    iter->age = age;
        
    if(age<0)
      iter = accumulatedObstaclesWorld.erase(iter);
    else
      ++iter;
  }

  // Adding new obstacles to accumulated map
  for(size_t i = 0; i < newObstaclesWithAgesWorld.size(); i++) {
    accumulatedObstaclesWorld.push_back(newObstaclesWithAgesWorld[i]);  
  } 

    
  // Plotting resulting accumulated obstacles in self frame for further filtering
  cv::Mat accumulatedObstaclesImage(X_MAX-X_MIN, Y_MAX-Y_MIN, CV_8UC1);  
  accumulatedObstaclesImage = 0;
  for(size_t i = 0; i < accumulatedObstaclesWorld.size(); i++) {
    Eigen::Vector3d ps = worldToSelf * accumulatedObstaclesWorld[i].posWorld;
    //Eigen::Vector3d ps = pw;
    int px = ps.x()/SUBCUBE_SIZE_M;
    int py = ps.y()/SUBCUBE_SIZE_M;
    double age = accumulatedObstaclesWorld[i].age;

    int row = (X_MAX-px);
    int col = (Y_MAX-py);
    if( (col >= 0)&&(col < accumulatedObstaclesImage.cols)&&(row >= 0)&&(row < accumulatedObstaclesImage.rows)) {
      accumulatedObstaclesImage.at<unsigned char>(row, col) = age; // Setting intensity of a point according for it's age for verbose
    }
  }

  // Clustering and filtering resulting accumulated obstacles map
  std::vector<std::vector<cv::Point> > accumulatedHulls;
  std::vector<cv::RotatedRect> accumulatedMinRects;    
  clusterAndFilterObstaclesByConnectedComponents(accumulatedObstaclesImage, accumulatedHulls, accumulatedMinRects);  
  
  // Exporting polygon obstacles for TebLocalPlanner, TaggedImg visualisation and classic rhoban stack
  detectedObstacles.clear();
  for( int i = 0; i< accumulatedMinRects.size(); i++ )   {

    double obstacle_center_x_in_m = (Y_MAX - accumulatedMinRects[i].center.y) * SUBCUBE_SIZE_M;
    double obstacle_center_y_in_m = (X_MAX - accumulatedMinRects[i].center.x) * SUBCUBE_SIZE_M;
    StereoObstacle obstacle;
    obstacle.pos_in_self = Eigen::Vector2d(obstacle_center_x_in_m, obstacle_center_y_in_m);
    obstacle.radius = (accumulatedMinRects[i].size.width + accumulatedMinRects[i].size.height) / 2.0;
    for( int j = 0; j < accumulatedHulls[i].size(); j++ ) {
      double hull_pt_x_in_m = (double)(X_MAX - accumulatedHulls[i][j].y) * SUBCUBE_SIZE_M;
      double hull_pt_y_in_m = (double)(Y_MAX - accumulatedHulls[i][j].x) * SUBCUBE_SIZE_M;        
      obstacle.hull_points_in_self.push_back(Eigen::Vector2d(hull_pt_x_in_m, hull_pt_y_in_m));
    }
    detectedObstacles.push_back(obstacle);    
  }

  // 7. Drawing verbose image
  cv::Mat obstaclesVerbose;
  std::vector<cv::Mat> channels;
  cv::Mat z = cv::Mat::zeros(accumulatedObstaclesImage.rows, accumulatedObstaclesImage.cols, CV_8UC1);
  channels.push_back(z);
  channels.push_back(freeSpacesHeightMap);
  channels.push_back(accumulatedObstaclesImage);  
  merge(channels, obstaclesVerbose);
  //cv::cvtColor(accumulatedObstaclesImage, obstaclesVerbose, cv::COLOR_GRAY2BGR);
  cv::line(obstaclesVerbose, cv::Point(obstaclesVerbose.cols/2, 0), cv::Point(obstaclesVerbose.cols/2, obstaclesVerbose.rows), cv::Scalar(128,128,128), 1);
  cv::line(obstaclesVerbose, cv::Point(0, obstaclesVerbose.rows/2), cv::Point(obstaclesVerbose.cols, obstaclesVerbose.rows/2), cv::Scalar(128,128,128), 1);

  //Drawing accumulated obstacles hulls/ellipses/minrects on verbose image
  for( int i = 0; i< accumulatedMinRects.size(); i++ ) {
    //Drawing hulls
    cv::polylines(obstaclesVerbose, accumulatedHulls[i], true, cv::Scalar(0,128,255), 1);        
    /*
    //Drawing rotated rectangles
    cv::Point2f rect_points[4]; newFrameMinRects[i].points( rect_points );
    for( int j = 0; j < 4; j++ ) cv::line( obstaclesVerbose, rect_points[j], rect_points[(j+1)%4], cv::Scalar(0,0,255), 1, 8 );
    */
    /*
    //Drawing ellipses inscribed in rotatedRects
    cv::ellipse( obstaclesVerbose, newFrameMinRects[i], cv::Scalar(0,0,255), 2, 8 );    
    */
  }

  cv::pyrUp(obstaclesVerbose, obstaclesVerbose);
  img() = obstaclesVerbose; 
}

void ObstacleMap::clusterAndFilterObstaclesByConnectedComponents(cv::Mat obstaclesHeightMap, std::vector<std::vector<cv::Point> > &hulls, std::vector<cv::RotatedRect> &minRects)
{

  cv::Mat obstaclesMask(X_MAX-X_MIN, Y_MAX-Y_MIN, CV_8UC1);
  threshold(obstaclesHeightMap, obstaclesMask,1,255,THRESH_BINARY);

  cv::Mat labelImage(obstaclesMask.size(), CV_32S);
  int nLabels = connectedComponents(obstaclesMask, labelImage, 8);
  
  //Naive rotated bbox calculation using opencv.
  //TODO: replace this with frustum calculation
  std::vector<std::vector<cv::Point> > contours;
  contours.resize(nLabels-1); //label=0 is always backround, we don't need it

  for(int y = 0; y < obstaclesMask.rows; y++) {
    for(int x = 0; x < obstaclesMask.cols; x++) {
      int label = labelImage.at<int>(y, x);
      if(label>0) contours[label-1].push_back(cv::Point(x,y));
    }
  }
  for( int i = 0; i < contours.size(); i++ ) {
    // //Calculating lowes height of a cluster to filter "flying blobs"
    // int min_height = 255;
    // for(int j=0; j<contours[i].size(); j++ ) {
    //   unsigned char val = obstaclesHeightMap.at<unsigned char>(contours[i][j].y, contours[i][j].x);
    //   if (val < min_height) min_height = val;
    // }
    // if(min_height > 128 + 50) continue;

    cv::RotatedRect minRect = minAreaRect( Mat(contours[i]) );
    float minSize = 0.75*(robocup_referee::Constants::field.ball_radius / SUBCUBE_SIZE_M);
    if(std::min(minRect.size.width, minRect.size.height)>minSize) { //Here shoud me max, not min. But min works better in real life
      minRects.push_back(minRect);
      std::vector<cv::Point> hull;
      convexHull( contours[i], hull );
      hulls.push_back(hull);
    }
  }
}

}  // namespace Filters
}  // namespace Vision

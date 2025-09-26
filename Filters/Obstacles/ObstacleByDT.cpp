#include "Filters/Obstacles/ObstacleByDT.hpp"
#include "CameraState/CameraState.hpp"
#include "robocup_referee/constants.h"
#include <opencv2/imgproc/imgproc.hpp>


#define DOWNSCALE 0
#define VERBOSE 1
using namespace std;
using namespace cv;

namespace Vision
{
namespace Filters
{

ObstacleByDT::ObstacleByDT() : Filter("ObstacleByDT"){
  
}

void ObstacleByDT::setParameters()
{
  /*
  left = ParamInt(0, 0, 10000, ParameterType::PARAM);
  top = ParamInt(0, 0, 10000, ParameterType::PARAM);
  width = ParamInt(0, 0, 10000, ParameterType::PARAM);
  height = ParamInt(0, 0, 10000, ParameterType::PARAM);
  params()->define<ParamInt>("left", &left);
  params()->define<ParamInt>("top", &top);
  params()->define<ParamInt>("width", &width);
  params()->define<ParamInt>("height", &height);
  */
}

double ObstacleByDT::my_fast_norm(double ax, double ay, double bx, double by) {
  return sqrt((ax - bx)*(ax - bx) + (ay - by)*(ay - by));
}

void ObstacleByDT::process()
{

  //clearObstaclesData();
  clearRobots();
  // Get names of dependencies
  const std::string & greenName = _dependencies[0];
  const std::string & widthProviderName = _dependencies[1];
  const std::string & yProviderName = _dependencies[2];
  // Import source matrix and update size
  const cv::Mat & src_green  = *(getDependency(greenName).getImg());
  const cv::Mat & widthImg = *(getDependency(widthProviderName).getImg());
  const cv::Mat & yImg = *(getDependency(yProviderName).getImg());

  Vision::Utils::CameraState *cs = &getCS();

  #if DOWNSCALE
  cv::Mat src;
  pyrDown(src_green, src);
  #else
  cv::Mat src = src_green.clone();
  #endif

  erode(src, src, Mat(), Point(-1, -1), 1, 1, 1);
  erode(src, src, Mat(), Point(-1, -1), 1, 1, 1);
  //erode(src, src, Mat(), Point(-1, -1), 1, 1, 1);
  
  dilate(src, src, Mat(), Point(-1, -1), 1, 1, 1);
  dilate(src, src, Mat(), Point(-1, -1), 1, 1, 1);
  //dilate(img(), img(), Mat(), Point(-1, -1), 1, 1, 1);  

  Mat dist;
  distanceTransform(255-src, dist, DIST_L1, 3); //DIST_L1 is faster than DIST_L2 with neglible quality degradation

  Mat verbose; //, verbose_gray;
  //dist.convertTo(verbose_gray, CV_8UC1);
  //cvtColor(verbose_gray, verbose, CV_GRAY2BGR);
  #if VERBOSE
    #if DOWNSCALE
    Mat yImg_small;
    pyrDown(yImg, yImg_small);
    cvtColor(yImg_small, verbose, CV_GRAY2BGR);
    #else
    cvtColor(yImg, verbose, CV_GRAY2BGR);
    #endif
  #endif

  Mat ff_mask = Mat(src.rows+2, src.cols+2, CV_8UC1);

  //Make ff_mask have non-zero values only on green areas and all non-green areas which width/height less than half of a ball
  //(so white lines are being eliminated here and won't distorb the result)
  //Here we assume that opponent robot is an object which have some part greater than ball in size and it's not green   
  for(int y=0; y < src.rows; y++) {
    for(int x=0; x < src.cols; x++) {
      #if DOWNSCALE
      float ball_width = widthImg.at<float>(y*2,x*2)/2.0;
      if((ball_width<2.5)||(dist.at<float>(y,x) < ball_width*0.5) ) ff_mask.at<unsigned char>(y,x) = 255;
      else ff_mask.at<unsigned char>(y,x) = 0;      
      #else
      float ball_width = widthImg.at<float>(y,x);
      if((ball_width<5)||(dist.at<float>(y,x) < ball_width*0.5) ) ff_mask.at<unsigned char>(y,x) = 255;
      else ff_mask.at<unsigned char>(y,x) = 0;      
      #endif

      //verbose.at<Vec3b>(y,x) = Vec3b(0,ff_mask.at<char>(y,x), 0);
    }
  }  



  //verbose.setTo(Vec3b(0,100,0), ff_mask(Rect(1,1, ff_mask.cols-2, ff_mask.rows-2)) );
  //verbose.setTo(0);

  int objects_count = 0;
  std::vector<cv::Rect> objects;
  Mat this_object_mask;

  Mat all_objects = Mat(src.rows+2, src.cols+2, CV_8UC1);
  all_objects = 0;
  
  for(int y=1; y < src.rows-1; y++) {
    for(int x=1; x < src.cols-1; x++) {
      
      //We need to check only holes in ff_mask
      if(ff_mask.at<unsigned char>(y+1,x+1)==0) {

        #if DOWNSCALE
        float ball_width = widthImg.at<float>(y*2,x*2)/2.0;
        #else
        float ball_width = widthImg.at<float>(y,x);
        #endif

        //if(dist.at<float>(y,x) > ball_width*0.8) {
        if(dist.at<float>(y,x) > ball_width*1.2) {
          //circle(verbose, Point2i(x,y), ball_width*1.2/2.0, Vec3b(255,0,0),-1);
          float loDiff = 255;
          float upDiff = 255;

          //floodfill stops when hit non-zero pixel in mask
          this_object_mask = ff_mask.clone();
          cv::Rect br;

          //This fllodfill will only fill the hole in this_object_mask and set the pixels filled to 255 in this_object_mask
          floodFill(dist, this_object_mask, Point2i(x,y), 0, &br, loDiff, upDiff, 4 | (255 << 8) | CV_FLOODFILL_MASK_ONLY | CV_FLOODFILL_FIXED_RANGE );

          //So the pixels setted to 255 in this_object_mask is our object
          this_object_mask.setTo(0, ff_mask);

          //Clear other pixels in mask to let only this object pixels survive
          ff_mask.setTo(255, this_object_mask);

          int cx = br.x + br.width/2;
          int cy = br.y + br.height/2;

          #if DOWNSCALE
          float ball_width_br_center = widthImg.at<float>(cy*2,cx*2)/2.0;
          #else
          float ball_width_br_center = widthImg.at<float>(cy,cx);
          #endif          

          //Check if this object have at least one dimension singnificantly larger than ball 
          if( (br.width > ball_width_br_center*3.0) || (br.height > ball_width_br_center*3.0)) {

            objects_count++;

            Mat all_objects_roi = all_objects(br);
            all_objects_roi.setTo(255, this_object_mask(br)); //TODO: maybe trim roi according to mask which is 2 pixels larger here?

            //br.x = cx - ball_width_br_center/2;
            //br.y = cy - ball_width_br_center/2;
            //br.width = ball_width_br_center;
            //br.height = ball_width_br_center;            
            objects.push_back(br);
          }

          //Vec3b color = Vec3b(rand()%255, rand()%255, rand()%255);
          //verbose.setTo( color, this_object_mask(Rect(1,1, this_object_mask.cols-2, this_object_mask.rows-2)) );
        }
      }
    }
  }
  //std::cout << "objects_count=" << objects_count << std::endl;

  //verbose.setTo( Vec3b(0,255,255), all_objects(Rect(1,1, all_objects.cols-2, all_objects.rows-2)) );

  cv::Point2f self_pos_on_img;
  bool self_pos_on_img_valid = false;
  try {
    Eigen::Vector3d self_zero_pos = cs->getWorldFromSelf(Eigen::Vector3d(.25, 0, 0));
    //self_pos_on_img = cs->imgXYFromWorldPositionWideangle(self_zero_pos);
    self_pos_on_img = cs->imgXYFromWorldPosition(self_zero_pos, Vision::Utils::CAMERA_WIDE_FULL);
    self_pos_on_img_valid = true;
    //std::cout << self_pos_on_img << std::endl;
    #if DOWNSCALE
    self_pos_on_img = self_pos_on_img / 2.0;
    #endif
    circle(verbose, self_pos_on_img, 10, Vec3b(255,0,255), -1);
  } catch (const std::runtime_error& exc) {
    //do nothing
  } 

  vector<vector<Point>> all_objects_contours;
  findContours( all_objects(Rect(1,1, all_objects.cols-2, all_objects.rows-2)),
                all_objects_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

  for(size_t i=0; i < all_objects_contours.size();i++) {
    vector<Point> this_contour = all_objects_contours[i];
    #if VERBOSE
    drawContours(verbose, all_objects_contours, i,  Vec3b(0,255,255), 1);
    #endif

    if(self_pos_on_img_valid) {
      double closest_dist = 10000;
      int closest_j = 0;
      for(size_t j=0;j<this_contour.size();j++) {
        double dist = my_fast_norm(this_contour[j].x, this_contour[j].y, self_pos_on_img.x, self_pos_on_img.y);
        if(dist < closest_dist) {
          closest_dist = dist;
          closest_j = j;
        }
      } 

      Point2f pt = Point2f(this_contour[closest_j].x, this_contour[closest_j].y);
      #if DOWNSCALE
      pt = pt*2.0;
      #endif

      Eigen::Vector3d obstacle_in_world = cs->posInWorldFromPixel(pt, 0.0, Vision::Utils::CAMERA_WIDE_FULL);
      Eigen::Vector3d obstacle_in_self = cs->getSelfFromWorld(obstacle_in_world);
      
      //Use only obstacles in front of the robot
      if(obstacle_in_self[0]>0.25) {
        #if DOWNSCALE
        float ball_width = widthImg.at<float>(pt*2.0);
        #else
        float ball_width = widthImg.at<float>(pt);
        #endif
        #if VERBOSE
        cv::line(verbose, self_pos_on_img, this_contour[closest_j], Vec3b(0,0,255), 2);
        circle(verbose, this_contour[closest_j], ball_width, Vec3b(0,0,255), 2);
        #endif
        //std::cout << "obstacle_in_self" << obstacle_in_self << std::endl;        
        //pushObstacle(this_contour[closest_j].x, this_contour[closest_j].y, verbose);

        //Keep only obstacles close enough 
        //if(my_fast_norm(obstacle_in_self[0],obstacle_in_self[1], 0.25,0) < robocup_referee::Constants::field.field_width) {
        if(my_fast_norm(obstacle_in_self[0],obstacle_in_self[1], 0.25,0) < robocup_referee::Constants::field.field_width/2) {
          pushRobot(Point2f(this_contour[closest_j].x, this_contour[closest_j].y));
        }
      }
    }
  }

  #if VERBOSE
  for( cv::Rect r : objects) {
    rectangle(verbose, r, Vec3b(255,255,255), 1);
  }
  #endif

 
    //Mat dist;
  //Mat voronoi;
  //distanceTransform(255-src, dist, voronoi, DIST_L2, 3, CV_DIST_LABEL_PIXEL);
  //img() = voronoi.clone();

  //normalize(dist, dist, 0, 255.0, NORM_MINMAX);
  //img() = dist.clone();  
  //img().convertTo(img(), CV_8UC1);
  //verbose.setTo( Vec3b(0,255,255), this_object_mask(Rect(1,1, this_object_mask.cols-2, this_object_mask.rows-2)) );
  #if VERBOSE
  img() = verbose;
  #else
  img() = 0;
  #endif
  
}
}  // namespace Filters
}  // namespace Vision

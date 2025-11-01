#include "Filters/Pipeline.hpp"
#include "Filters/Custom/WhiteLinesBirdview.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "CameraState/CameraState.hpp"
#include "robocup_referee/constants.h"
#include <rhoban_utils/util.h>
#include "rhoban_utils/timing/benchmark.h"
#include <rhoban_utils/logging/logger.h>
#include "RhIO.hpp"
//#include "fit_ellipse.h"

#define USE_LINES 1
#define USE_CORNERS 1

#define PARALLEL_THREADS 8
#define USE_PARALLEL_FOR 1

//#include <opencv2/ximgproc.hpp>

/*
  Json usage example (in clipping.json):
    {
        "class name" : "WhiteLines",
        "content" : {
            "name" : "whiteLinesDetector",
            "display" : true,
            "dependencies" : ["Y", "integralY", "greenHSV"]

        }
    }
  integralY shold be raw Y integral transform, not yNoRobot as is default 2019 env (!)
 */

struct CornerCandiate
{
  //All in normalised coords (0..1 span on x and y)
  cv::Point2f ufa;
  cv::Point2f vfa;
  cv::Point2f ufb;
  cv::Point2f vfb;
  cv::Point2f corner;
  double score;
  bool valid;
};

using namespace std;
using namespace cv;
using robocup_referee::Constants;    

namespace Vision {
namespace Filters {

WhiteLinesBirdview::WhiteLinesBirdview() : Filter("WhiteLinesBirdview") {
  frame_n = 0;
  myInitRhIO();
}

WhiteLinesBirdview::~WhiteLinesBirdview() {}
  
void WhiteLinesBirdview::setParameters() {
}


double WhiteLinesBirdview::my_norm(cv::Point2f a, cv::Point2f b) {
  return sqrt( (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) );
}

// Returns 1 if the lines intersect, otherwise 0. In addition, if the lines 
// intersect the intersection point may be stored in the floats i_x and i_y.
char WhiteLinesBirdview::get_line_intersection(float p0_x, float p0_y, float p1_x, float p1_y, 
    float p2_x, float p2_y, float p3_x, float p3_y, float *i_x, float *i_y)
{
    float s1_x, s1_y, s2_x, s2_y;
    s1_x = p1_x - p0_x;     s1_y = p1_y - p0_y;
    s2_x = p3_x - p2_x;     s2_y = p3_y - p2_y;

    float s, t;
    float ds,dt;
    ds = (-s2_x * s1_y + s1_x * s2_y);
    dt = (-s2_x * s1_y + s1_x * s2_y);
    
    if((ds!=0)&&(dt!=0)) {

	    s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / ds;
	    t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / dt;

	    if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
	    {
		// Collision detected
		if (i_x != NULL)
		    *i_x = p0_x + (t * s1_x);
		if (i_y != NULL)
		    *i_y = p0_y + (t * s1_y);
		return 1;
	    }
    }

    return 0; // No collision
}

inline int WhiteLinesBirdview::getRegionSum(Mat isum, int x, int y, int w, int h) {

      int tl= isum.at<int>(y,x);
      int tr= isum.at<int>(y,x+w);
      int bl= isum.at<int>(y+h,x);
      int br= isum.at<int>(y+h,x+w);
      return br-bl-tr+tl;
}

void WhiteLinesBirdview::non_maxima_suppression(const cv::Mat& image, cv::Mat& mask, int sizex, int sizey, int threshold) {
    // find pixels that are equal to the local neighborhood not maximum (including 'plateaus')
    Mat kernel = getStructuringElement(cv::MORPH_RECT, Size(sizex,sizey));
    //cv::dilate(image, mask, cv::Mat());
    cv::dilate(image, mask, kernel);
    for(int y=0; y < image.rows; y++) {
      for(int x=0; x < image.cols; x++) {
        if((image.at<uchar>(y,x) < mask.at<uchar>(y,x)) || (image.at<uchar>(y,x)<threshold) ) mask.at<uchar>(y,x) = 0;
        else mask.at<uchar>(y,x) = 255;
      }
    }
}

void WhiteLinesBirdview::get_plane_equation(double x1, double y1, double z1, 
                        double x2, double y2, double z2,  
                        double x3, double y3, double z3,
                        double *ra, double *rb, double *rc, double *rd) 
{ 
  //taken from https://www.geeksforgeeks.org/program-to-find-equation-of-a-plane-passing-through-3-points/
    double a1 = x2 - x1; 
    double b1 = y2 - y1; 
    double c1 = z2 - z1; 
    double a2 = x3 - x1; 
    double b2 = y3 - y1; 
    double c2 = z3 - z1; 
    double a,b,c,d;
    a = b1 * c2 - b2 * c1; 
    b = a2 * c1 - a1 * c2; 
    c = a1 * b2 - b1 * a2; 
    d = (- a * x1 - b * y1 - c * z1); 
    *ra = a;
    *rb = b;
    *rc = c;
    *rd = d;
} 


float WhiteLinesBirdview::get_line2line_angle(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4) //in radians
{
	float a = x1 - x2;
	float b = y1 - y2;
	float c = x3 - x4;
	float d = y3 - y4;
	//
	float cos_angle , angle;
	float mag_v1 = sqrt(a*a + b*b);
	float mag_v2 = sqrt(c*c + d*d);
	if((mag_v1 * mag_v2)!=0) {
		cos_angle = (a*c + b*d) / (mag_v1 * mag_v2);
		if(fabs(cos_angle)>1) return 0;
		angle = acos(cos_angle);
		return angle;
	}
	return 0;
}


//Rewritten from https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
float WhiteLinesBirdview::get_line_magnitude(float x1, float y1, float x2, float y2) {
  //Get line (aka vector) length'
  return sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
}

float WhiteLinesBirdview::get_point2line_distance(float px, float py, float x1, float y1, float x2, float y2) {
  //Get distance between point and line
  //http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba

  float LineMag = get_line_magnitude(x1, y1, x2, y2);
  if (LineMag < 0.00000001) return 9999; //line are too short for proper computations

  float u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)));
  float u = u1 / (LineMag * LineMag);

  if ((u < 0.00001) || (u > 1)) {
    // point does not fall within the line segment, take the shorter distance to an endpoint
    float ix = get_line_magnitude(px, py, x1, y1);
    float iy = get_line_magnitude(px, py, x2, y2);
    return std::min(ix,iy);
  } else {
    //Intersecting point is on the line, use the formula
    float ix = x1 + u * (x2 - x1);
    float iy = y1 + u * (y2 - y1);
    return get_line_magnitude(px, py, ix, iy);
  }
}

float WhiteLinesBirdview::get_segnent2segment_distance(float xa1, float ya1, float xa2, float ya2, float xb1, float yb1, float xb2, float yb2) {
  //Get all possible distances between each dot of two lines and second line and return the shortest
  float dist1 = get_point2line_distance(xa1,ya1, xb1, yb1, xb2, yb2);
  float dist2 = get_point2line_distance(xa2,ya2, xb1, yb1, xb2, yb2);
  float dist3 = get_point2line_distance(xb1,yb1, xa1, ya1, xa2, ya2);
  float dist4 = get_point2line_distance(xb2,yb2, xa1, ya1, xa2, ya2);

  return std::min(std::min(dist1, dist2), std::min(dist3, dist4));
}

void WhiteLinesBirdview::merge_two_segments(float xa1, float ya1, float xa2, float ya2, 
                        float xb1, float yb1, float xb2, float yb2, 
                        float *xr1, float *yr1, float *xr2, float *yr2) 
{
  //Dumb merging of two segments by removing two points closest to centroid
  float centroid_x = (xa1+xa2+xb1+xb2)/4.0;
  float centroid_y = (ya1+ya2+yb1+yb2)/4.0;
  float dist[4];
  dist[0] = get_line_magnitude(xa1,ya1, centroid_x, centroid_y);
  dist[1] = get_line_magnitude(xa2,ya2, centroid_x, centroid_y);
  dist[2] = get_line_magnitude(xb1,yb1, centroid_x, centroid_y);
  dist[3] = get_line_magnitude(xb2,yb2, centroid_x, centroid_y);

  float max_dist1 = 0;
  int max_dist1_index = 0;
  for(int i=0;i<4;i++) {
    if(dist[i]>max_dist1) {
      max_dist1 = dist[i];
      max_dist1_index = i;      
    }
  }
  //std::cout << "max_dist1_index=" << max_dist1_index << std::endl;
  if(max_dist1_index==0) { *xr1 = xa1; *yr1 = ya1; }
  if(max_dist1_index==1) { *xr1 = xa2; *yr1 = ya2; }
  if(max_dist1_index==2) { *xr1 = xb1; *yr1 = yb1; }
  if(max_dist1_index==3) { *xr1 = xb2; *yr1 = yb2; }

  dist[max_dist1_index] = 0; //to skip this point in next iteration  
  float max_dist2 = 0;
  int max_dist2_index = 0;
  for(int i=0;i<4;i++) {
    if(dist[i]>max_dist2) {
      max_dist2 = dist[i];
      max_dist2_index = i;      
    }
  }  
  if(max_dist2_index==0) { *xr2 = xa1; *yr2 = ya1; }
  if(max_dist2_index==1) { *xr2 = xa2; *yr2 = ya2; }
  if(max_dist2_index==2) { *xr2 = xb1; *yr2 = yb1; }
  if(max_dist2_index==3) { *xr2 = xb2; *yr2 = yb2; }

  //std::cout << "Merging " << cv::Point2f(xa1,ya1) << "-" << cv::Point2f(xa2,ya2) << " and " << cv::Point2f(xb1,yb1) << "-" << cv::Point2f(xb2,yb2) << " to " << cv::Point2f(*xr1,*yr1) << "-" << cv::Point2f(*xr2,*yr2) << std::endl;
}

void WhiteLinesBirdview::get_circle(cv::Point2f p1,cv::Point2f p2,cv::Point2f p3, cv::Point2f& center, float& radius)
{
  double x1 = p1.x;
  double x2 = p2.x;
  double x3 = p3.x;

  double y1 = p1.y;
  double y2 = p2.y;
  double y3 = p3.y;

  radius = 0;

  center.x = (x1*x1+y1*y1)*(y2-y3) + (x2*x2+y2*y2)*(y3-y1) + (x3*x3+y3*y3)*(y1-y2);
  double divx = ( 2*(x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2) );
  if(divx==0) return;
  center.x /= divx;

  center.y = (x1*x1 + y1*y1)*(x3-x2) + (x2*x2+y2*y2)*(x1-x3) + (x3*x3 + y3*y3)*(x2-x1);
  double divy = ( 2*(x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2) );
  if(divy==0) return;
  center.y /= divy;

  radius = sqrt((center.x-x1)*(center.x-x1) + (center.y-y1)*(center.y-y1));
}

int WhiteLinesBirdview::get_circle_score(const std::vector<cv::Point2f> ransacInput, cv::Point2f center, float radius) 
{
  float targetRadius = Constants::field.center_radius;
  if( fabs(radius-targetRadius)/targetRadius>0.1) return 0; //circle radius more than 10% different from central circle radius from constants
  //std::cout << "Verifying circle: center=" << center << ", radius=" << radius << std::endl;

  int score = 0;
  /*for(int i=0;i<ransacInput.size();i++) {
    double dist_from_circle_center = my_norm(ransacInput[i], center);
    if(fabs(dist_from_circle_center - radius) < 0.05) score++;
  }
  return score;*/
  int counter = 0;
  for(float t = 0; t<2.0*M_PI; t+= 2.0*M_PI/180.0)
  {
    counter++;
    float cX = radius*cos(t) + center.x;
    float cY = radius*sin(t) + center.y;
    bool inlier_found = false;
    for (int j=0;j<ransacInput.size();j++) {
      double d =  my_norm(ransacInput[j], cv::Point2f(cX, cY));
      if(d<0.05/2.0) {
        inlier_found = true;
        break;
      }
    }
    if(inlier_found) score++;
  }
  return score;
}

//RANSAC for circle in real worlds's coords
int WhiteLinesBirdview::ransac_circle(std::vector<cv::Point2f> ransacInput, cv::Point2f& center, float& radius)
{
    
    cv::Point2f bestCircleCenter;
    float bestCircleRadius;
    int best_circle_inliers = 0;

    int maxNrOfIterations = ransacInput.size();

    for(unsigned int its=0; its< maxNrOfIterations; ++its)
    {
        // randomly choose 3 points:
        unsigned int idx1 = rand() % ransacInput.size();
        unsigned int idx2 = rand() % ransacInput.size();
        unsigned int idx3 = rand() % ransacInput.size();

        // we need 3 different samples:
        if(idx1 == idx2) continue;
        if(idx1 == idx3) continue;
        if(idx3 == idx2) continue;

        if( my_norm(ransacInput[idx1], ransacInput[idx2]) < 0.05) continue;
        if( my_norm(ransacInput[idx1], ransacInput[idx3]) < 0.05) continue;
        if( my_norm(ransacInput[idx2], ransacInput[idx3]) < 0.05) continue;
        

        // create circle from 3 points:
        cv::Point2f c; 
        float r;
        get_circle(ransacInput[idx1], ransacInput[idx2], ransacInput[idx3], c, r);
                
        //verify or falsify the circle by inlier counting:
        int inliers = get_circle_score(ransacInput, c, r);

        // update best circle information if necessary
        if(inliers >= best_circle_inliers)
        {
            best_circle_inliers = inliers;
            bestCircleRadius = r;
            bestCircleCenter = c;
        }
    }

    if(best_circle_inliers > 0) {
      center = bestCircleCenter;
      radius = bestCircleRadius;
      return best_circle_inliers;
    }
    return 0;
}


#if USE_PARALLEL_FOR

class ParallelIngegralImageXTraverser : public ParallelLoopBody
{
  private:
    Mat &m_iresx;
    const Mat &m_isum;
    const int m_im_cols;
    const int m_win_line_width;
    const int m_win_height;
    const int m_scanline_divider;

  public:
    ParallelIngegralImageXTraverser (Mat &iresx, const Mat &isum, const int im_cols, const int win_line_width, const int win_height, const int scanline_divider)
        : m_iresx(iresx), m_isum(isum), m_im_cols(im_cols), m_win_line_width(win_line_width), m_win_height(win_height), m_scanline_divider(scanline_divider)
    {
    }
    
    inline int m_getRegionSum(const Mat &isum, int x, int y, int w, int h) const {

      int tl= isum.at<int>(y,x);
      int tr= isum.at<int>(y,x+w);
      int bl= isum.at<int>(y+h,x);
      int br= isum.at<int>(y+h,x+w);
      return br-bl-tr+tl;
    }

    virtual void operator ()(const Range& range) const 
    {
      //std::cout << "range[" << range.start << ".." << range.end << "]" << std::endl;
      for (int yn = range.start; yn < range.end; yn++)
      {
        int y = yn*m_scanline_divider;
        for(int x=0; x < m_im_cols-1; x++) {
          if (x>=m_im_cols-m_win_line_width*3-1) break; //not to hit the border
          int sleft   = m_getRegionSum(m_isum, x,y, m_win_line_width, m_win_height);
          int smiddle = m_getRegionSum(m_isum, x+m_win_line_width, y, m_win_line_width, m_win_height);
          int sright  = m_getRegionSum(m_isum, x+m_win_line_width*2, y, m_win_line_width, m_win_height);
          int diff = (smiddle-sleft)*(smiddle-sright)/(m_win_line_width*m_win_height)/(m_win_line_width*m_win_height)/20;
          if((smiddle-sright)<0) diff=0;
          if((smiddle-sleft)<0) diff=0;
          if(diff<0) diff = 0;
          if(diff>255) diff = 255;
          m_iresx.ptr<uchar>(y+m_win_height/2)[x+(m_win_line_width*3)/2] = diff;
        }
      }
    }
};

class ParallelIngegralImageYTraverser : public ParallelLoopBody
{
  private:
    Mat &m_iresy;
    const Mat &m_isum;
    const int m_im_rows;
    const int m_win_line_width;
    const int m_win_height;
    const int m_scanline_divider;

  public:
    ParallelIngegralImageYTraverser (Mat &iresy, const Mat &isum, const int im_rows, const int win_line_width, const int win_height, const int scanline_divider)
        : m_iresy(iresy), m_isum(isum), m_im_rows(im_rows), m_win_line_width(win_line_width), m_win_height(win_height), m_scanline_divider(scanline_divider)
    {
    }
    
    inline int m_getRegionSum(const Mat &isum, int x, int y, int w, int h) const {

      int tl= isum.at<int>(y,x);
      int tr= isum.at<int>(y,x+w);
      int bl= isum.at<int>(y+h,x);
      int br= isum.at<int>(y+h,x+w);
      return br-bl-tr+tl;
    }

    virtual void operator ()(const Range& range) const 
    {
      //std::cout << "range[" << range.start << ".." << range.end << "]" << std::endl;
      for (int xn = range.start; xn < range.end; xn++)
      {
        int x = xn*m_scanline_divider;
        for(int y=0; y < m_im_rows-1; y++) {
          if (y>=m_im_rows-m_win_line_width*3-1) break; //not to hit the border          
          int sleft   = m_getRegionSum(m_isum, x,y, m_win_height, m_win_line_width);
          int smiddle = m_getRegionSum(m_isum, x,y+m_win_line_width, m_win_height, m_win_line_width);
          int sright  = m_getRegionSum(m_isum, x,y+m_win_line_width*2, m_win_height, m_win_line_width);
          int diff = (smiddle-sleft)*(smiddle-sright)/(m_win_line_width*m_win_height)/(m_win_line_width*m_win_height)/20;
          if((smiddle-sright)<0) diff=0;
          if((smiddle-sleft)<0) diff=0;          
          if(diff<0) diff = 0;
          if(diff>255) diff = 255;
          m_iresy.ptr<uchar>(y+(m_win_line_width*3)/2)[x+m_win_height/2] = diff;
        }
      }
    }
};
#endif
                                                                                                          

void WhiteLinesBirdview::process() {

  //return;

  myInitRhIO();
  importPropertiesFromRhIO();

  Benchmark::open("Converting input image");

  Vision::Utils::CameraState *cs = &getCS();

  std::string sourceName = _dependencies[0]; 
  cv::Mat source = (getDependency(sourceName).getImg())->clone();

  std::string integralYName = _dependencies[1]; 
  cv::Mat integralY = (getDependency(integralYName).getImg())->clone();

  std::string greenName = _dependencies[2]; 
  cv::Mat green = (getDependency(greenName).getImg())->clone();  

  int row_nb = source.rows;
  int col_nb = source.cols;
  //img() = cv::Mat(row_nb, col_nb, CV_8UC3);

  //Defining verbose image
  cv::Mat im(row_nb, col_nb, CV_8UC3, cv::Scalar(0,0,0));
  
  //Copy source gray image to verbose output
  cv::cvtColor(source, im, CV_GRAY2BGR);
  
  
  //Mark green filter output on verbose image for debug
  for(int y=0; y < im.rows; y++) {
    for(int x=0; x < im.cols; x++) {
      if(green.at<uchar>(y,x)>0) {
        Vec3b clr = im.at<Vec3b>(y,x);
        im.at<Vec3b>(y,x) = Vec3b(clr(0)/2, clr(1), clr(2)/2);
      }
    }
  }
  



  //To debug input image:
  //source.copyTo(img());
  //return;

  //Definig clipping image
  cv::Mat fieldOnly = cv::Mat(row_nb, col_nb, CV_8UC1);
  fieldOnly.setTo(255);

	Benchmark::close("Converting input image");

  //Clipping by horizon - for birdview should be done prior to perspective transform, i.e. not here

  Mat isum = integralY; //Pick ready integral image from rhoban's pipeline
  //Mat isum; cv::integral(source, isum); //Calculate integral image by ourself 

  Mat iresx = Mat(im.rows, im.cols, CV_8UC1);
  Mat iresy = Mat(im.rows, im.cols, CV_8UC1);
  Mat iresrgb = Mat(im.rows, im.cols, CV_8UC3);
  iresx.setTo(0);
  iresy.setTo(0);
  iresrgb.setTo(0);

  
  Benchmark::open("Integral image traversing");

  //Widnow size orthogonal to traverse direction
  const int win_height=5; 

  //Assumed width of lines to be detected (in pixels)
  const int win_line_width = 10; //For birdview, DIRTY HARDCODE 

  //Divider to check only Nth scanline. Greatly improves speed with almost no quality degradation. 4 seems to be enoughth for everybody (c)
  const int scanline_divider = 4;

  //Now we can traverse image in x and y direction and do some integral image processing using window with 3 parts - left part, middle part, right part. 
  //Each part have width equal to assumed line width on a birdview

  //x traverse/scalnine
  
  #if USE_PARALLEL_FOR
    ParallelIngegralImageXTraverser parallelIngegralImageXTraverser(iresx, isum, im.cols, win_line_width, win_height, scanline_divider);
    parallel_for_(Range(0, (im.rows-win_height)/scanline_divider), parallelIngegralImageXTraverser, PARALLEL_THREADS);
  #else
    for(int yn=0; yn < (im.rows-win_height)/scanline_divider; yn++) {
        int y = yn*scanline_divider;
          for(int x=0; x < im.cols-1; x++) {
            if (x>=im.cols-win_line_width*3-1) break; //not to hit the border
            int sleft   = getRegionSum(isum, x,y, win_line_width, win_height);
            int smiddle = getRegionSum(isum, x+win_line_width, y, win_line_width, win_height);
            int sright  = getRegionSum(isum, x+win_line_width*2, y, win_line_width, win_height);
            int diff = (smiddle-sleft)*(smiddle-sright)/(win_line_width*win_height)/(win_line_width*win_height)/20;
            if((smiddle-sright)<0) diff=0;
            if((smiddle-sleft)<0) diff=0;
            if(diff<0) diff = 0;
            if(diff>255) diff = 255;
            iresx.at<uchar>(y+win_height/2, x+(win_line_width*3)/2) = diff;
        }
    }
  #endif
  
  //y traverse/scanline
  #if USE_PARALLEL_FOR
    ParallelIngegralImageYTraverser parallelIngegralImageYTraverser(iresy, isum, im.rows, win_line_width, win_height, scanline_divider);
    parallel_for_(Range(0, (im.cols-win_height)/scanline_divider), parallelIngegralImageYTraverser, PARALLEL_THREADS);
  #else  
    for(int xn=0; xn < (im.cols-win_height)/scanline_divider; xn++) {
        int x = xn*scanline_divider;
          for(int y=0; y < im.rows-1; y++) {
            if (y>=im.rows-win_line_width*3-1) break; //not to hit the border          
            int sleft   = getRegionSum(isum, x,y, win_height, win_line_width);
            int smiddle = getRegionSum(isum, x,y+win_line_width, win_height, win_line_width);
            int sright  = getRegionSum(isum, x,y+win_line_width*2, win_height, win_line_width);
            int diff = (smiddle-sleft)*(smiddle-sright)/(win_line_width*win_height)/(win_line_width*win_height)/20;
            if((smiddle-sright)<0) diff=0;
            if((smiddle-sleft)<0) diff=0;          
            if(diff<0) diff = 0;
            if(diff>255) diff = 255;
            iresy.at<uchar>(y+(win_line_width*3)/2,x+win_height/2) = diff;
        }
    }
  #endif
  

  Benchmark::close("Integral image traversing");

   std::vector<cv::Point2f> circle_ransac_input;

  Benchmark::open("Non-maxima suppression");
  
  //Now let's do non-maxima suppression on traverse results - this will thin results to single pixel width in the middle of detected line (if line quality is not too bad)
  Mat nmsx = cv::Mat(row_nb, col_nb, CV_8UC1);
  Mat nmsy = cv::Mat(row_nb, col_nb, CV_8UC1);

  //Suppress all pixels except local maximas with intensity thresold
  //int line_intencity_treshold = 7;  //Was 5 for old narrow-angle processing
  non_maxima_suppression(iresx, nmsx, win_line_width*3, 1, lineIntensityTreshold);
  non_maxima_suppression(iresy, nmsy, 1, win_line_width*3, lineIntensityTreshold);

  Benchmark::close("Non-maxima suppression");


  Benchmark::open("Green check after NMS");
  //Resulting mat for both x and y results in single channel
  Mat nms = cv::Mat(row_nb, col_nb, CV_8UC1);
  nms.setTo(0);

  //Checking for proper green color at left and right sides of a line
  for(int y=0; y < im.rows-win_height; y++) {
    for(int x=0; x < im.cols; x++) {
      if(nmsx.at<uchar>(y,x)>0) {        
        int val = 255;
        //Mark a line pixel blue before green check
        im.at<Vec3b>(y,x) = Vec3b(255, 0, 0);
        im.at<Vec3b>(y-1,x) = Vec3b(255, 0, 0);
        im.at<Vec3b>(y+1,x) = Vec3b(255, 0, 0);
        im.at<Vec3b>(y,x-1) = Vec3b(255, 0, 0);
        im.at<Vec3b>(y,x+1) = Vec3b(255, 0, 0); 

        //Filling data vector for central circle RANSAC
        circle_ransac_input.push_back(cv::Point2f(x,y));

        if (x >= im.cols-win_line_width*3/2-1) break; //not to hit the border
        if (x < win_line_width*3/2) break; //not to hit the border

        int gleft = countNonZero(green(Rect(x-(win_line_width*3)/2,y, win_line_width,win_height)));
        //int gmiddle = countNonZero(green(Rect(x-(win_line_width*3)/2+win_line_width,y, win_line_width,win_height)));
        int gright = countNonZero(green(Rect(x-(win_line_width*3)/2+win_line_width*2,y, win_line_width,win_height)));

        if(gleft < 0.3*(float)win_line_width*(float)win_height) val = 0;  //30% green coverage to the left of the line, TODO: remove hardcode 
        if(gright < 0.3*(float)win_line_width*(float)win_height) val = 0;  //30% green coverage to the right of the line, TODO: remove hardcode   
        //if(gmiddle > 0.5*(float)win_line_width*(float)win_height) val = 0; //Works bad so disabled (green filter can mark white lines as 99% green when its painted bad)
        if(val>0) nms.at<uchar>(y,x) = val;
      }
    }
  }

  for(int x=0; x < im.cols-win_height; x++) {
    for(int y=0; y < im.rows; y++) {
      if(nmsy.at<uchar>(y,x)>0) {

        //Mark a line pixel blue before green check
        im.at<Vec3b>(y,x) = Vec3b(255, 0, 0);
        im.at<Vec3b>(y-1,x) = Vec3b(255, 0, 0);
        im.at<Vec3b>(y+1,x) = Vec3b(255, 0, 0);
        im.at<Vec3b>(y,x-1) = Vec3b(255, 0, 0);
        im.at<Vec3b>(y,x+1) = Vec3b(255, 0, 0);

        int val = 255;
        if (y >= im.rows-win_line_width*3/2-1) break; //not to hit the border 
        if (y < win_line_width*3/2-1) break; //not to hit the border 

        int gleft = countNonZero(green(Rect(x,y-win_line_width*3/2, win_height,win_line_width)));
        //int gmiddle = countNonZero(green(Rect(x,y-(win_line_width*3)/2+win_line_width, win_height,win_line_width)));
        int gright = countNonZero(green(Rect(x,y-win_line_width*3/2+win_line_width*2, win_height,win_line_width)));       
        if(gleft < 0.3*(float)win_line_width*(float)win_height) val = 0;  //30% green coverage to the left of the line, TODO: remove hardcode 
        if(gright < 0.3*(float)win_line_width*(float)win_height) val = 0;  //30% green coverage to the right of the line, TODO: remove hardcode   
        //if(gmiddle > 0.5*(float)win_line_width*(float)win_height) val = 0; //Works bad so disabled (green filter can mark white lines as 99% green when its painted bad)    
        if(val>0) nms.at<uchar>(y,x) = val;
      }

    }
  }
   


  //For debug verbose
  int show_linedetector_heatmap = 0;
  if(show_linedetector_heatmap) {
    im.setTo(0);
    for(int y=0; y < iresrgb.rows; y++) {
      for(int x=0; x < iresrgb.cols; x++) {
        //im.at<Vec3b>(y,x) = Vec3b(source.at<uchar>(y,x), iresy.at<uchar>(y,x), iresx.at<uchar>(y,x));
        im.at<Vec3b>(y,x) = Vec3b(0, iresy.at<uchar>(y,x), iresx.at<uchar>(y,x));
        //im.at<Vec3b>(y,x) = Vec3b(nms.at<uchar>(y,x), nms.at<uchar>(y,x), nms.at<uchar>(y,x)); //Displays black image with proper line pixels marked wgite
      }
    }  

    im.copyTo(img());
    if(0) { //Save B/W points as images for outsorce RANSAC tests         
      char s[255];
      sprintf(s, "/tmp/img%04d.jpg", frame_n);
      cv:imwrite(s, im);
      frame_n++;
    }

    Benchmark::close("Green check after NMS");
    return;
  } else {
    for(int y=0; y < iresrgb.rows; y++) {
      for(int x=0; x < iresrgb.cols; x++) {
        if(nms.at<uchar>(y,x)>0) {
          //im.at<Vec3b>(y,x) = Vec3b(0, 0, 255); //Mark a line pixel with proper green by red pixel
          im.at<Vec3b>(y,x) = Vec3b(0, 0, 255);
          im.at<Vec3b>(y-1,x) = Vec3b(0, 0, 255);
          im.at<Vec3b>(y+1,x) = Vec3b(0, 0, 255);
          im.at<Vec3b>(y,x-1) = Vec3b(0, 0, 255);
          im.at<Vec3b>(y,x+1) = Vec3b(0, 0, 255);
          
          //Filling data vector for central circle RANSAC
          //circle_ransac_input.push_back(cv::Point2f(x,y));
          //cv::Point2f pc = cs->robotPosFromImg(x, y, source.cols, source.rows);
          /*cv::Point2f pc;
          try {
            pc = cs->robotPosFromImg(x, y);
          } catch (const std::runtime_error& exc) {
            continue;
          }            
          circle_ransac_input.push_back(pc);*/
        }
      }
    }      
  }

  Benchmark::close("Green check after NMS");
  
  //Ransac ellipse for central circle
  /*if(circle_ransac_input.size()>5) {
    RotatedRect rel = fit_ellipse(circle_ransac_input, iresx, iresy);
    ellipse(im, rel, Vec3b(0,255,255), 2);
    //std::cout << "=================RotatedRect=" << rel << std::endl;
  }
  */

  //Ransac for central circle
  /*
  if(circle_ransac_input.size()>3) {
    int ransac_circle_inliers = 0;
    cv::Point2f circle_center;
    float circle_radius;
    ransac_circle_inliers =  ransac_circle(circle_ransac_input, circle_center, circle_radius);
    if(ransac_circle_inliers>(float)180*0.25) {
      //good circle is a circle with percentage of inliers more than 50%, 
      //or more than 25% inliers and more than 50% of points not in FOV
      int points_in_fov = 0;
      std::vector<cv::Point> circle_verbose_points;
      for(float t = 0; t<2.0*M_PI; t+= 6.0*M_PI/180.0) //total 60 points per circle
       {
          float rX = circle_radius*cos(t) + circle_center.x;
          float rY = circle_radius*sin(t) + circle_center.y;
          //TODO: imgXYFromRobotPosition is very time consuming
          cv::Point cp = cs->imgXYFromRobotPosition(cv::Point2f(rX, rY), source.cols, source.rows);
          if( (cp.x<source.cols) && (cp.x>0) && (cp.y<source.rows) && (cp.y>0) ) {
            circle_verbose_points.push_back(cp);
            points_in_fov++;
          }
      }
      if( (ransac_circle_inliers>(float)180*0.5) || ( (ransac_circle_inliers>(float)180*0.25)&&(points_in_fov<(float)60*0.5) ) ) {
        std::cout << "Found circle: center=" << circle_center << ", radius=" << circle_radius << ", inliers=" << ransac_circle_inliers <<", points_in_fov=" << points_in_fov << std::endl;
        for(int i=0;i<circle_verbose_points.size();i++) {
          //circle(im, circle_verbose_points[i], 2, Scalar(255,0,255), -1);  
          //im.at<cv::Vec3b>(circle_verbose_points[i].y, circle_verbose_points[i].x) = cv::Vec3b(255,0,255);
          cv::line(im, circle_verbose_points[i], circle_verbose_points[(i+1)%circle_verbose_points.size()], cv::Scalar(255,0,255), 2, LINE_AA);
        }
        cv::Point cc = cs->imgXYFromRobotPosition(circle_center, source.cols, source.rows);
        circle(im, cc, 10, Scalar(255,0,255), 2);  
      }
    }
  }
  */

 
  Benchmark::open("Probabilistic Hough Transform");

  //Probabilistic Line Transform after NMS and green check
  dilate(nms, nms, Mat(), Point(-1, -1), 1, 1, 1);
  vector<Vec4i> linesP_all;
  int pointsThreshold = 100/scanline_divider;
  int lengthThreshold = 50;
  int gapThreshold = 15; //25 from 2018 version gives too musk false positives;
  HoughLinesP(nms, linesP_all, 2, 2*CV_PI/180, pointsThreshold, lengthThreshold, gapThreshold );

  //Extending lines length a little bit for more robust intersecion detection
  //For birdview its safe to use win_line_width as an extension size - we won't join goalzone corner with border line
  double extend_in_pix = win_line_width;
  for( size_t i = 0; i < linesP_all.size(); i++ )
  {
      Vec4i l = linesP_all[i];
      double len = sqrt(pow(l[2]-l[0], 2) + pow(l[3]-l[1], 2));
      double ldx = (l[2]-l[0])/len;
      double ldy = (l[3]-l[1])/len;
      linesP_all[i][2] += ldx * extend_in_pix;
      linesP_all[i][0] -= ldx * extend_in_pix;
      linesP_all[i][3] += ldy * extend_in_pix;
      linesP_all[i][1] -= ldy * extend_in_pix;
  }

  Benchmark::close("Probabilistic Hough Transform");

  Benchmark::open("Merging segments");
  //Merging similar segments
  //TODO: Not effective and leave small lines as artifacts, need to be rewritten (!)
  bool changes_done_line = true;
  while(changes_done_line) {
    changes_done_line = false;
    for( size_t i = 0; i < linesP_all.size(); i++ ) {
      //for( size_t j = i+1; j < linesP_all.size(); j++ ) {
      for( size_t j = 0; j < linesP_all.size(); j++ ) {        
        if( (i!=j) 
         && (get_line_magnitude(linesP_all[i][0], linesP_all[i][1], linesP_all[i][2], linesP_all[i][3])>1.0) 
         && (get_line_magnitude(linesP_all[j][0], linesP_all[j][1], linesP_all[j][2], linesP_all[j][3])>1.0) ) {

          if( (get_segnent2segment_distance(linesP_all[i][0], linesP_all[i][1], linesP_all[i][2], linesP_all[i][3],
                                          linesP_all[j][0], linesP_all[j][1], linesP_all[j][2], linesP_all[j][3])<10.0)

            &&(fabs(get_line2line_angle(linesP_all[i][0], linesP_all[i][1], linesP_all[i][2], linesP_all[i][3],
                                  linesP_all[j][0], linesP_all[j][1], linesP_all[j][2], linesP_all[j][3]))*180.0/M_PI<5.0) ) {
            //This segments should be merged
            float mx1, my1, mx2, my2;
            merge_two_segments( linesP_all[i][0], linesP_all[i][1], linesP_all[i][2], linesP_all[i][3],
                                linesP_all[j][0], linesP_all[j][1], linesP_all[j][2], linesP_all[j][3],
                                &mx1, &my1, &mx2, &my2);
            linesP_all[j][0] = mx1;
            linesP_all[j][1] = my1;
            linesP_all[j][2] = mx2;
            linesP_all[j][3] = my2;
            linesP_all[i][0] = 0;
            linesP_all[i][1] = 0;
            linesP_all[i][2] = 0;
            linesP_all[i][3] = 0;
            changes_done_line = true;          
          }
        }
      }
    }
  }

  Benchmark::close("Merging segments");

  Benchmark::open("Drawing segments");

  vector<Vec4i> linesP;
  for( size_t i = 0; i < linesP_all.size(); i++ )
  {
    if(get_line_magnitude(linesP_all[i][0], linesP_all[i][1], linesP_all[i][2], linesP_all[i][3])>1.0) {
      linesP.push_back(linesP_all[i]);  
    }
  }
  //std::cout << "Segments merge results: before=" << linesP_all.size() <<", after=" << linesP.size() << std::endl;



  // Draw merged hough segments (without corner angle check) as thin yellow lines
  for( size_t i = 0; i < linesP.size(); i++ )
  {
      //Scalar color = Scalar(rand()%255,rand()%255,rand()%255 );
      Vec4i l = linesP[i];
      line( im, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,200,200), 1);
      //line( im, Point(l[0], l[1]), Point(l[2], l[3]), color, 1, LINE_AA);
  }
   
  Benchmark::close("Drawing segments");

  int piom = cs->getBirdviewPixelsInOneMeter();
  int width = im.cols;
  int height = im.rows;

  // Drawing distance circles eachcd 1m
  for (int i = 1; i < (1 + 2 * Constants::field.field_width); i++)
  {
    cv::circle(im, cv::Point2i(width / 2, height ), (double)i * piom, cv::Scalar(0, 0, 0), 1);
  }

  // 0° orientation (front)
  cv::line(im, cv::Point2i(width / 2, height), cv::Point2f(width / 2, 0), cv::Scalar(0, 0, 0), 1);

  
  Benchmark::open("Checking lines for intersections");
	//Check for intersections and fill an array of corner candidates
  double largestCornerLength = 0;
  std::vector<CornerCandiate> cornerCandidates;  
  float maxDist = std::max(Constants::field.field_length + Constants::field.border_strip_width_x*2, Constants::field.field_width + Constants::field.border_strip_width_x*2);
  for( size_t i = 0; i < linesP.size(); i++ )
  {
		for( size_t j = i+1; j < linesP.size(); j++ )
	  {
		  float i_x, i_y;
			Vec4i la = linesP[i];
			Vec4i lb = linesP[j];
      char intersect = get_line_intersection(la[0], la[1], la[2], la[3], 
                                            lb[0], lb[1], lb[2], lb[3],
                                            &i_x, &i_y);
			if(intersect) {
        cv::Point2f ps0, ps1, ps2, ps3, pi;
        try
        {
          ps0 = cs->robotPosFromBirdviewImg(la[0], la[1]);
          ps1 = cs->robotPosFromBirdviewImg(la[2], la[3]);
          ps2 = cs->robotPosFromBirdviewImg(lb[0], lb[1]);
          ps3 = cs->robotPosFromBirdviewImg(lb[2], lb[3]);
          pi = cs->robotPosFromBirdviewImg(i_x, i_y);
          //ps0 = cv::Point2f(la[0], la[1])/10.0;
          //ps1 = cv::Point2f(la[2], la[3])/10.0;
          //ps2 = cv::Point2f(lb[0], lb[1])/10.0;
          //ps3 = cv::Point2f(lb[2], lb[3])/10.0;
          //pi = cv::Point2f(i_x, i_y)/10.0;          
        }
          catch (const std::runtime_error& exc)
        {
          continue;
        }
        float length_a = my_norm(ps0,ps1);
        float length_b = my_norm(ps2,ps3);
				if( (ps0.x < maxDist)&&(ps0.y < maxDist)&&(ps1.x < maxDist)&&(ps1.y < maxDist)&& //intersection point is farther than field size
				    (ps2.x < maxDist)&&(ps2.y < maxDist)&&(ps3.x < maxDist)&&(ps3.y < maxDist)&& //intersection point is farther than field size
				    ( sqrt(pow(pi.x,2)+ pow(pi.y,2)) >0.2 )&&  //Also skip here 0.2 meters radius zone around robot to ignore handling bar, etc
            (length_a>0.3 && length_b>0.3) ) //filter out penalty mark, TODO: remove hardcode*/
        //if(1)
        {
					float thetaDeg = get_line2line_angle(ps0.x, ps0.y, ps1.x, ps1.y, ps2.x, ps2.y, ps3.x, ps3.y) * 180.0/ M_PI;
					float deltaDeg = 90.0 - fabs(thetaDeg);
					if(fabs(deltaDeg)<20.0) { //Allow max 20 degrees misalignment from ideal 90 degrees intersection

            if(length_a + length_b > largestCornerLength) {
              largestCornerLength = length_a + length_b;
              
              largestCornerAngleDrift = fabs(deltaDeg);   
            }

            cv::Point2f Ufa(((float) la[0]), ((float) la[1])); //2019
            cv::Point2f Vfa(((float) la[2]), ((float) la[3])); //2019
            cv::Point2f Ufb(((float) lb[0]), ((float) lb[1])); //2019
            cv::Point2f Vfb(((float) lb[2]), ((float) lb[3])); //2019     
            cv::Point2f C(((float) i_x), ((float) i_y));       //2019     

            CornerCandiate cornerCandidate;
            cornerCandidate.ufa = Ufa;
            cornerCandidate.vfa = Vfa;
            cornerCandidate.ufb = Ufb;
            cornerCandidate.vfb = Vfb;            
            cornerCandidate.corner = C;
            cornerCandidate.score = my_norm(Ufa, Vfa) + my_norm(Ufb, Vfb);
            cornerCandidate.valid = true;
            cornerCandidates.push_back(cornerCandidate);

					}
				}
			}

		}
  }

  Benchmark::close("Checking lines for intersections");

  Benchmark::open("Merging corners");
  //Merge close corners by keeping only one corner in a cluster with largest line lengths. Not optimal but works
  bool changes_done = true;
  while(changes_done) {
    changes_done = false;
    for( size_t i = 0; i < cornerCandidates.size(); i++ ) {
      for( size_t j = i+1; j < cornerCandidates.size(); j++ ) {
        cv::Point2f c1 =  cornerCandidates[i].corner; 
        cv::Point2f c2 =  cornerCandidates[j].corner;
        if((cornerCandidates[i].valid)&&(cornerCandidates[j].valid)) {
          if(my_norm(c1,c2)<0.1) {
            //Corners are too close, filter one of them
            if(cornerCandidates[i].score>cornerCandidates[j].score) cornerCandidates[j].valid = false;
            else cornerCandidates[i].valid = false;
            changes_done = true;
          }
        }
      }
    }
  }  
  Benchmark::close("Merging corners");

  

  Benchmark::open("Filling data for robocup.cpp");
  //Data exchange vector for robocup.cpp
  loc_data_vector.clear();

  #if USE_LINES
  //First stage - fill data exchange vector with filtered line candidates (not corners)
  for( size_t i = 0; i < linesP.size(); i++ ) {
    Vec4i l = linesP[i];

    //Add only lines with appropriate length.
    //This will filter central circle segments and short sides of goal zone
    //std::cout << "Found line length = " << my_norm(p0,p1) << "m" << std::endl;
    float min_pixel_length_on_birdview = cs->getBirdviewPixelsInOneMeter()*1.5;
    float min_pixel_length_on_wideangle_full = 175;
    float min_real_length = Constants::field.goal_area_length*1.5;

    //Determining real coords of line start/end (in m)
    cv::Point2f p0, p1;
    try {
      p0 = cs->robotPosFromBirdviewImg(l[0], l[1]);
      p1 = cs->robotPosFromBirdviewImg(l[2], l[3]);
    }
      catch (const std::runtime_error& exc)
    {
      std::cout << "Can't do robotPosFromBirdviewImg on Line start/end" << std::endl;
      continue;
    }

    //Determining pixel coords of line start/end on a source wideangle image (not birdview), 
    //to filter short false lines found on opponent robots, etc which become "good long" lines after birdview transform

    //Don't work because of frequent "point outsude the theoretical image" error 
    //TODO: bring it back and check on [ln -sf ~/starkitrobots/workspace/env/starkit3_spinnaker_wideangle/manual_logs/rhobanLegs/ workingLog]

    /*Eigen::Vector3d pos_self_p0 = Eigen::Vector3d(p0.x, p0.y, 0.0);
    Eigen::Vector3d pos_self_p1 = Eigen::Vector3d(p1.x, p1.y, 0.0);
    cv::Point2f wideangle_pixel_pos_p0, wideangle_pixel_pos_p1;
    
    double pixel_length_on_wideangle_full;

    //try {    
      wideangle_pixel_pos_p0 = cs->imgXYFromSelf(pos_self_p0, Vision::Utils::CAMERA_WIDE_FULL);
      wideangle_pixel_pos_p1 = cs->imgXYFromSelf(pos_self_p1, Vision::Utils::CAMERA_WIDE_FULL);
      pixel_length_on_wideangle_full = my_norm(wideangle_pixel_pos_p0, wideangle_pixel_pos_p1); 
    //}
    //  catch (const std::runtime_error& exc)
    //{
    //  std::cout << "Can't do imgXYFromSelf on Line start/end" << std::endl;
    //  //continue;
    //  pixel_length_on_wideangle_full = 1000; //Fake large length
    //}  
    */ 


    //std::cout << "p0=" << p0 << std::endl;
    //std::cout << "p1=" << p1 << std::endl;
    //std::cout << "my_norm=" << my_norm(p0, p1) << std::endl;
    if( my_norm(p0,p1) < min_real_length) continue;

    //std::cout << "birdview_p0=" << cv::Point2f(l[0], l[1]) << std::endl;
    //std::cout << "birdview_p1=" << cv::Point2f(l[2], l[3]) << std::endl;
    //std::cout << "my_norm=" << my_norm(cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3])) << std::endl;
    if( my_norm(cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3])) < min_pixel_length_on_birdview) continue;

    //std::cout << "wideangle_pixel_pos_p0=" << wideangle_pixel_pos_p0 << std::endl;
    ///std::cout << "wideangle_pixel_pos_p1=" << wideangle_pixel_pos_p1 << std::endl;
    //std::cout << "pixel_length_on_wideangle_full=" << pixel_length_on_wideangle_full << std::endl;
    //if( pixel_length_on_wideangle_full < min_pixel_length_on_wideangle_full) continue;

    cv::Point2f Uf(((float) l[0]), ((float) l[1])); //2019    
    cv::Point2f Vf(((float) l[2]), ((float) l[3])); //2019

    //Creatin WhiteLinesData object, it can contains one or two lines and zero or one corner. 
    //For now add only lines
    float dummy_quality = 1.0;
    WhiteLinesData whiteLinesData = WhiteLinesData();    
    whiteLinesData.max_dist_corner = 10.0;
    whiteLinesData.tolerance_angle_line = 20.0; //10.0;
    whiteLinesData.tolerance_angle_corner = 25.0; //15.0; 
    whiteLinesData.minimal_segment_length = 0.001; //0.3;
    if(l[1]>l[3]) whiteLinesData.pushPixLine(dummy_quality, std::pair<cv::Point2f,cv::Point2f>(Uf,Vf));
    else whiteLinesData.pushPixLine(dummy_quality, std::pair<cv::Point2f,cv::Point2f>(Vf,Uf));
    loc_data_vector.push_back(whiteLinesData);

    //Draw the line being pushed to localisation in bold yellow
    line( im, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,255,255), 4, LINE_AA);
  }
  #endif

  #if USE_CORNERS
  //Second stage - fill data exchange vector with filtered corner candidates
  for( size_t i = 0; i < cornerCandidates.size(); i++ ) {
    if(cornerCandidates[i].valid) {

      //For whiteLinesData.cpp only Ufa-C-Vfb points is needed
      cv::Point2f Ufa;
      cv::Point2f Vfb;
      cv::Point2f C;

      checkRefineAndDrawCornerCandidate(im, cornerCandidates[i].ufa, cornerCandidates[i].vfa, cornerCandidates[i].corner, &Ufa, &Vfb, &C);
      checkRefineAndDrawCornerCandidate(im, cornerCandidates[i].ufa, cornerCandidates[i].vfb, cornerCandidates[i].corner, &Ufa, &Vfb, &C);
      checkRefineAndDrawCornerCandidate(im, cornerCandidates[i].ufb, cornerCandidates[i].vfb, cornerCandidates[i].corner, &Ufa, &Vfb, &C);
      checkRefineAndDrawCornerCandidate(im, cornerCandidates[i].ufb, cornerCandidates[i].vfa, cornerCandidates[i].corner, &Ufa, &Vfb, &C);


      
      
      /*
      //For now localisation module accepts only L-shaped corners.
      //Angle from "a" line to "b" line should be clockwise
      //So let's convert found corner (it can be X-shape, L-shape or T-shape) to L-shape with proper line order
      cv::Point2f Ufa = cornerCandidates[i].ufa;
      cv::Point2f Vfa = cornerCandidates[i].vfa;
      cv::Point2f Ufb = cornerCandidates[i].ufb;
      cv::Point2f Vfb = cornerCandidates[i].vfb;      
      cv::Point2f C = cornerCandidates[i].corner;  

      //Cutting lengths of shortest segments in T/X-spaed corner to convert it to L-shaped 
      //Corner will be processed as Ua-C-Vb notation in WhiteLiesCornerobservation.cpp, so Ub/Va values are useless
      float norm_a1 = my_norm(Ufa,C);
      float norm_a2 = my_norm(Vfa,C);
      if(norm_a1>norm_a2) Vfa = C; else { Ufa = Vfa; Vfa = C; };

      float norm_b1 = my_norm(Ufb,C);
      float norm_b2 = my_norm(Vfb,C);
      if(norm_b2>norm_b1) Ufb = C; else { Vfb = Ufb; Ufb = C; };
      
      //Angle from Ufa-C vector to Vfb-C vector should be anti-clockwise, 
      //overwise calculations of corner orientation in WhiteLiesCornerobservation.cpp will be wrong.
      //So let's correct each corner by cross product (=vector multiplication) z-component sign check
      cv::Point2f a = Ufa-C;
      cv::Point2f b = Vfb-C;
      double cross_product_z_component = a.x * b.y - b.x * a.y;
      if(cross_product_z_component>0) {
        //We need to swap Ufa and Vfb
        cv::Point2f t;
        t = Vfb;
        Vfb = Ufa;
        Ufa = t;
      }
      //Drawing raw corner (without conversion to L-shape) in bold green
      line( im, Point(cornerCandidates[i].ufa.x, cornerCandidates[i].ufa.y), 
                Point(cornerCandidates[i].vfa.x, cornerCandidates[i].vfa.y), Scalar(0,255,0), 1);
      line( im, Point(cornerCandidates[i].ufb.x, cornerCandidates[i].ufb.y), 
                Point(cornerCandidates[i].vfb.x, cornerCandidates[i].vfb.y), Scalar(0,255,0), 1);                                
      //circle(im, Point(cornerCandidates[i].corner.x, cornerCandidates[i].corner.y), 5, Scalar(0,0,255), -1);
      //Drawing triangle base for better corner visibility
      line( im, Point(Ufa.x, Ufa.y), 
                Point(Vfb.x, Vfb.y), Scalar(125,255,0), 1); 

      //Drawing corner with conversion to L-shape
      //Angle from red to pink should be clockwise
      line( im, Point(Ufa.x, Ufa.y), 
                Point(C.x, C.y), Scalar(0,0,255), 2);
      line( im, Point(Vfb.x, Vfb.y), 
                Point(C.x, C.y), Scalar(0,128,255), 2);                
            
      //Creating WhiteLinesData object, it can contains one or two lines and zero or one corner. Only corners is used by now
      float dummy_quality = 1.0;
      WhiteLinesData whiteLinesData = WhiteLinesData();        
      whiteLinesData.max_dist_corner = 10.0;
      whiteLinesData.tolerance_angle_line = 20.0; //10.0;
      whiteLinesData.tolerance_angle_corner = 25.0; //15.0; 
      whiteLinesData.minimal_segment_length = 0.001; //0.3;
      whiteLinesData.pushPixLine(dummy_quality, std::pair<cv::Point2f,cv::Point2f>(Ufa,Vfa));
      whiteLinesData.pushPixLine(dummy_quality, std::pair<cv::Point2f,cv::Point2f>(Ufb,Vfb));
      whiteLinesData.addPixCorner(C); //This will automatically set has_corner to true
      
      loc_data_vector.push_back(whiteLinesData);
      */
    }
  }
  #endif

  Benchmark::close("Filling data for robocup.cpp");

  

  
 

  //Strategic TODO: one more pass of HoughP with shorter lines and strict length and angle check for penaty mark detection?

  //Strategic TODO: dewarp points and do RANSAC circle for field center detection

  
  // Drawing horizon for debug
  /*
  //TODO: bring it back in 2019 release
  cv::Point horizon1, horizon2;
  horizon1.x = 0;
  horizon2.x = source.cols - 1;
  horizon1.y = cs->getPixelYtAtHorizon(horizon1.x, source.cols, source.rows);
  horizon2.y = cs->getPixelYtAtHorizon(horizon2.x, source.cols, source.rows);
  cv::line(im, horizon1, horizon2, cv::Scalar(128, 128, 0), 2);
  */
  
  Benchmark::open("computeTransformations"); 

  for(int i=0;i<loc_data_vector.size();i++) {
    loc_data_vector[i].computeTransformations(&getCS(), true);
  }
  Benchmark::close("computeTransformations");

 

  cv::Point2i unionSquareCenterPix = cs->getUnionSquareCenterOnBirdviewPix();
  line(im, unionSquareCenterPix+Point2i(-piom/2, piom/2), unionSquareCenterPix+Point2i(piom/2, piom/2), cv::Vec3b(0,0,255), 1);
  line(im, unionSquareCenterPix+Point2i(piom/2, piom/2), unionSquareCenterPix+Point2i(piom/2, -piom/2), cv::Vec3b(0,0,255), 1);
  line(im, unionSquareCenterPix+Point2i(piom/2, -piom/2), unionSquareCenterPix+Point2i(-piom/2, -piom/2), cv::Vec3b(0,0,255), 1);
  line(im, unionSquareCenterPix+Point2i(-piom/2, -piom/2), unionSquareCenterPix+Point2i(-piom/2, piom/2), cv::Vec3b(0,0,255), 1);


  img() = im;
  //im.copyTo(img());
  //std::cout << "-------------- WHITE LINES PROCRESS() EXIT ------------" << std::endl;


  myPublishToRhIO();
    
}


void WhiteLinesBirdview::checkRefineAndDrawCornerCandidate(cv::Mat &im, const cv::Point2f p0, const cv::Point2f p1, const cv::Point2f intersection, 
                                    cv::Point2f *Ufa, cv::Point2f *Vfb, cv::Point2f *C)
{
     
      /* //Angle from "a" line to "b" line should be clockwise
      //So let's convert found corner (it can be X-shape, L-shape or T-shape) to L-shape with proper line order
      cv::Point2f Ufa = cornerCandidates[i].ufa;
      cv::Point2f Vfa = cornerCandidates[i].vfa;
      cv::Point2f Ufb = cornerCandidates[i].ufb;
      cv::Point2f Vfb = cornerCandidates[i].vfb;      
      cv::Point2f C = cornerCandidates[i].corner;  

      //Cutting lengths of shortest segments in T/X-spaed corner to convert it to L-shaped 
      //Corner will be processed as Ua-C-Vb notation in WhiteLiesCornerobservation.cpp, so Ub/Va values are useless
      float norm_a1 = my_norm(Ufa,C);
      float norm_a2 = my_norm(Vfa,C);
      if(norm_a1>norm_a2) Vfa = C; else { Ufa = Vfa; Vfa = C; };

      float norm_b1 = my_norm(Ufb,C);
      float norm_b2 = my_norm(Vfb,C);
      if(norm_b2>norm_b1) Ufb = C; else { Vfb = Ufb; Ufb = C; };
      */

      //Angle from Ufa-C vector to Vfb-C vector should be anti-clockwise, 
      //overwise calculations of corner orientation in WhiteLiesCornerobservation.cpp will be wrong.
      //So let's correct each corner by cross product (=vector multiplication) z-component sign check
      cv::Point2f a = p0 - intersection;
      cv::Point2f b = p1 - intersection;
      double cross_product_z_component = a.x * b.y - b.x * a.y;
      if(cross_product_z_component>0) {
        //We need to swap Ufa and Vfb
        *Ufa = p1;
        *Vfb = p0;
        *C = intersection;
      } else {
        *Ufa = p0;
        *Vfb = p1;
        *C = intersection;     
      }

      float norm0 = my_norm(p0, intersection);
      float norm1 = my_norm(p1, intersection);

      double min_corner_segment_length_in_pixels = getCS().getBirdviewPixelsInOneMeter()*0.5;
      if( (norm0>min_corner_segment_length_in_pixels) && (norm1>min_corner_segment_length_in_pixels) ) {
        //Refined corner is vaid
                            
        circle(im, Point(C->x, C->y), 5, Scalar(0,0,255), -1);
        //Drawing triangle base for better corner visibility
        line( im, Point(Ufa->x, Ufa->y), Point(Vfb->x, Vfb->y), Scalar(125,255,0), 1); 

        //Drawing corner with conversion to L-shape
        //Angle from red to pink should be clockwise
        line( im, Point(Ufa->x, Ufa->y), Point(C->x, C->y), Scalar(0,0,255), 2);
        line( im, Point(Vfb->x, Vfb->y), Point(C->x, C->y), Scalar(0,128,255), 2);                
              
        //Creating WhiteLinesData object, it can contains one or two lines and zero or one corner. 
        //If it's a corner - only Ufa, Vfb and C points are important 
        float dummy_quality = 1.0;
        cv::Point2f dummy_point = cv::Point2f(0,0);
        WhiteLinesData whiteLinesData = WhiteLinesData();        
        whiteLinesData.max_dist_corner = 10.0;
        whiteLinesData.tolerance_angle_line = 20.0; //10.0;
        whiteLinesData.tolerance_angle_corner = 25.0; //15.0; 
        whiteLinesData.minimal_segment_length = 0.001; //0.3;
        whiteLinesData.pushPixLine(dummy_quality, std::pair<cv::Point2f,cv::Point2f>(*Ufa, *C));
        whiteLinesData.pushPixLine(dummy_quality, std::pair<cv::Point2f,cv::Point2f>(*C, *Vfb));
        whiteLinesData.addPixCorner(*C); //This will automatically set has_corner to true
        
        loc_data_vector.push_back(whiteLinesData);
        
      }


}

void WhiteLinesBirdview::myInitRhIO()
{
  std::string undistort_path = rhio_path + getName() + "/largestCornerAngleDrift";
  if (RhIO::Root.getValueType(undistort_path) != RhIO::ValueType::NoValue)
    return;
  RhIO::Root.newFloat(undistort_path)->defaultValue(largestCornerAngleDrift);
  RhIO::Root.newFloat(undistort_path)->defaultValue(largestCornerAngleDrift);
  
  std::string filter_path = rhio_path + getName();
  if (RhIO::Root.getValueType(filter_path) != RhIO::ValueType::NoValue)
    return;
  RhIO::IONode& node = RhIO::Root.child(filter_path);
  node.newFloat("lineIntensityTreshold")->defaultValue(30); //7 for a real world, 30 or even more for webots
}

void WhiteLinesBirdview::myPublishToRhIO()
{

  std::string undistort_path = rhio_path + getName() + "/largestCornerAngleDrift";
  RhIO::Root.setFloat(undistort_path, largestCornerAngleDrift); 
  //RhIO::Root.setFloat("/Vision/treatmentDelay", treatmentDelay);
  //RhIO::Root.setFloat("/Vision/lastUpdate", diffMs(lastTS, getNowTS()));
  //std::string cameraStatus = getCameraStatus();
  //RhIO::Root.setStr("/Vision/cameraStatus", cameraStatus);
}

void WhiteLinesBirdview::importPropertiesFromRhIO()
{
  std::string filter_path = rhio_path + getName();
  RhIO::IONode& node = RhIO::Root.child(filter_path);
  lineIntensityTreshold = node.getValueFloat("lineIntensityTreshold").value;
}

}
}

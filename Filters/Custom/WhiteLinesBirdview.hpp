#pragma once

#include "Filters/Filter.hpp"
#include "Filters/Custom/WhiteLinesData.hpp"


namespace Vision {
namespace Filters {

/**
 * Recognize Border of the Field and compute the clipping
 */
class WhiteLinesBirdview : public Filter {

 public:
  WhiteLinesBirdview();
  ~WhiteLinesBirdview();

  virtual std::string getClassName() const override { return "WhiteLinesBirdview"; }
  virtual int expectedDependencies() const override { return 3; }

  std::vector<WhiteLinesData> loc_data_vector;
  double lineIntensityTreshold;

 protected:
  /**
   * @Inherit
   */
  virtual void process() override;
  virtual void setParameters() override;

  private:
    int ransac_circle(std::vector<cv::Point2f> ransacInput, cv::Point2f& center, float& radius);
    int get_circle_score(const std::vector<cv::Point2f> ransacInput, cv::Point2f center, float radius);
    void get_circle(cv::Point2f p1,cv::Point2f p2,cv::Point2f p3, cv::Point2f& center, float& radius);
    void merge_two_segments(float xa1, float ya1, float xa2, float ya2, 
                        float xb1, float yb1, float xb2, float yb2, 
                        float *xr1, float *yr1, float *xr2, float *yr2); 
    float get_segnent2segment_distance(float xa1, float ya1, float xa2, float ya2, float xb1, float yb1, float xb2, float yb2);
    float get_point2line_distance(float px, float py, float x1, float y1, float x2, float y2);
    float get_line_magnitude(float x1, float y1, float x2, float y2);
    float get_line2line_angle(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4);
    void get_plane_equation(double x1, double y1, double z1, 
                        double x2, double y2, double z2,  
                        double x3, double y3, double z3,
                        double *ra, double *rb, double *rc, double *rd); 
    void non_maxima_suppression(const cv::Mat& image, cv::Mat& mask, int sizex, int sizey, int threshold);
    inline int getRegionSum(cv::Mat isum, int x, int y, int w, int h);
    char get_line_intersection(float p0_x, float p0_y, float p1_x, float p1_y, 
                              float p2_x, float p2_y, float p3_x, float p3_y, float *i_x, float *i_y);
    double my_norm(cv::Point2f a, cv::Point2f b);  


    void checkRefineAndDrawCornerCandidate(cv::Mat &im, const cv::Point2f p0, const cv::Point2f p1, const cv::Point2f intersection, 
                                    cv::Point2f *Ufa, cv::Point2f *Vfb, cv::Point2f *C);
    int frame_n; 

    double largestCornerAngleDrift;
    double largestCornerLength;
    void myInitRhIO();
    void myPublishToRhIO();
    void importPropertiesFromRhIO();

};

}
}

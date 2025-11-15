#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include "../camera_state/camera_state.hpp"
using cv::Mat;

// i do not know where it is defined, so
extern int lineIntensityTreshold;
extern double FIELD_WIDTH; // in meters
extern double FIELD_LENGTH; // in meters
extern int BORDER_STRIP_WIDTH_X;
extern double GOAL_AREA_WIDTH; // in meters

// Assumed width of lines to be detected (in pixels)
extern int LINE_WIDTH_BIRDVIEW;
// Widnow size orthogonal to traverse direction
extern int WINDOW_HEIGHT_BIRDVIEW;

inline int getRegionSum(Mat isum, int x, int y, int w, int h);
void non_maxima_suppression(
        const cv::Mat& image, 
        cv::Mat& mask, 
        int sizex, int sizey, 
        int threshold
);
float get_line_magnitude(
        float x1, float y1, 
        float x2, float y2
);
float get_point2line_distance(
        float px, float py, 
        float x1, float y1, 
        float x2, float y2
);
float get_segnent2segment_distance(
        float xa1, float ya1,
        float xa2, float ya2,
        float xb1, float yb1,
        float xb2, float yb2
);
float get_line2line_angle(
        float x1, float y1,
        float x2, float y2,
        float x3, float y3,
        float x4, float y4
);  // in radians

void merge_two_segments(
        float xa1, float ya1,
        float xa2, float ya2,
        float xb1, float yb1,
        float xb2, float yb2,
        float* xr1, float* yr1,
        float* xr2, float* yr2
);
char get_line_intersection(
        float p0_x, float p0_y,
        float p1_x, float p1_y,
        float p2_x, float p2_y,
        float p3_x, float p3_y,
        float *i_x, float *i_y
);
double my_norm(cv::Point2f a, cv::Point2f b);
void integral_image_traversal(Mat isum, Mat& iresx, Mat& iresy);
void tune_white_lines(Mat source, Mat integralY, Mat green, CameraState* cs);
void draw_meter_circles(Mat& im, int px_per_m);
void segment_white_lines(
        Mat source,
        Mat integralY,
        Mat green,
        CameraState* cs,
        const char* winname="white lines"
);

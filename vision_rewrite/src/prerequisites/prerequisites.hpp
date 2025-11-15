#include <opencv2/opencv.hpp>
using cv::Mat;

#define GREEN_FILTER_HUE_LOWERB 35
#define GREEN_FILTER_HUE_UPPERB 85
#define GREEN_FILTER_SAT_LOWERB 90
#define GREEN_FILTER_SAT_UPPERB 255

#define DEFAULT_ARUCO_DICT (cv::aruco::DICT_5X5_50)
#define ARUCO_SIZE 50
#define ARUCO_POS_X 700
#define ARUCO_POS_Y 1200

Mat integral_Y(Mat image);
Mat green_filter_HSV(
        Mat image, 
        int h_lowerb=GREEN_FILTER_HUE_LOWERB, int h_upperb=GREEN_FILTER_HUE_UPPERB,
        int s_lowerb=GREEN_FILTER_SAT_LOWERB, int s_upperb=GREEN_FILTER_SAT_UPPERB,
        int v_lowerb=GREEN_FILTER_SAT_LOWERB, int v_upperb=GREEN_FILTER_SAT_UPPERB
);

Mat get_birdview_from_aruco(Mat image, 
        std::vector<cv::Point2f> aruco_real_corners);
Mat get_perspective_tf_from_aruco(Mat image, 
        std::vector<cv::Point2f> aruco_real_corners);
Mat tune_rotation_and_position(const Mat& orig);
void transform_points(
            const std::vector<cv::Point2f>& pts,
            std::vector<cv::Point2f>& result,
            double angle,
            cv::Point2f center,
            double scale
);

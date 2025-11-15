#include "prerequisites.hpp"
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/highgui.hpp"

using cv::Mat;

Mat integral_Y(Mat image) {
    Mat YCbCr;
    cv::cvtColor(image, YCbCr, cv::COLOR_BGR2YUV);
    std::vector<Mat> YCbCr_channels(3);
    cv::split(YCbCr, YCbCr_channels);
    Mat integralY = YCbCr_channels[2];
    cv::integral(image, integralY);
    return integralY;
}

Mat green_filter_HSV(
        Mat image, 
        int h_from, int h_to,
        int s_from, int s_to,
        int v_from, int v_to
        ) {
    Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    std::vector<int> lowerb = {h_from, s_from, v_from};
    std::vector<int> upperb = {h_to, s_to, v_to};

    Mat green_mask;
    cv::inRange(hsv, lowerb, upperb, green_mask);
    return green_mask;
}


std::vector<std::vector<cv::Point2f>> detect_aruco(const Mat& img) {

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(DEFAULT_ARUCO_DICT);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);

    detector.detectMarkers(img, markerCorners, markerIds, rejectedCandidates);
    return markerCorners;
}

Mat get_perspective_tf_from_aruco(
            Mat image, 
            std::vector<cv::Point2f> aruco_real_corners
        ) {
    std::vector<std::vector<cv::Point2f>> markerCorners = detect_aruco(image);

    Mat perspective_tf = cv::getPerspectiveTransform(markerCorners[0], aruco_real_corners);
    return perspective_tf;
}

Mat get_birdview_from_aruco(Mat image, std::vector<cv::Point2f> aruco_real_corners) {
    Mat perspective_tf = get_perspective_tf_from_aruco(image, aruco_real_corners);
    Mat warped;
    cv::warpPerspective(image, warped, perspective_tf, {1000, 1000});
    return warped;
}

void transform_points(
            const std::vector<cv::Point2f>& pts,
            std::vector<cv::Point2f>& result,
            double angle,
            cv::Point2f center,
            double scale
        ) {
    result = pts;
    for (int i = 0; i < result.size(); ++i){
        result[i] += center;
    }
    Mat src(result);
    Mat dst;
    Mat tf = cv::getRotationMatrix2D(center, angle, scale);
    cv::transform(src, dst, tf);
    dst.copyTo(result);
}

Mat tune_rotation_and_position(const Mat& orig){
    const char winname[] = "tuning rot and pos";
    cv::namedWindow(winname);
    uint32_t img_h = orig.rows;
    uint32_t img_w = orig.cols;
    int x_pos = 594, y_pos = 884;
    int rot_int = 90;
    const int max_rot_tb = 360;

    int scale = 50;
    const int max_scale = 250;

    cv::createTrackbar("pos x", winname, &x_pos, img_w);
    cv::createTrackbar("pos y", winname, &y_pos, img_h);
    // cv::createTrackbar("scale", winname, &scale, max_scale);
    cv::createTrackbar("rot  ", winname, &rot_int, max_rot_tb);
    
    std::vector<cv::Point2f> base_aruco_points = {
        {0.5, 0.5},
        {-0.5, 0.5},
        {-0.5, -0.5},
        {0.5, -0.5}
    };
    std::vector<cv::Point2f> aruco_pts_tf;
    cv::Point2f center = {};
    Mat result_image;

    char key = 0;
    while (key != ' ') {
        // rotation = M_PI * 2 * rot_int / max_rot_tb;
        center = {(float)x_pos, (float)y_pos};
        transform_points(
                base_aruco_points, aruco_pts_tf, 
                rot_int, 
                center, 
                scale
        );
        result_image = get_birdview_from_aruco(orig, aruco_pts_tf);
        cv::imshow(winname, result_image);
        key = cv::waitKey(1);
    }
    cv::destroyWindow(winname);
    return result_image;
}


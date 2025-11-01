#include "prerequisites.hpp"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "opencv2/core/mat.hpp"

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
        int s_from, int s_to
        ) {
    Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    std::vector<int> lowerb = {h_from, s_from, 0};
    std::vector<int> upperb = {h_to, s_to, 255};

    Mat green_mask;
    cv::inRange(hsv, lowerb, upperb, green_mask);
    return green_mask;
}

Mat get_perspective_tf_from_aruco(Mat image) {
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

        cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
        cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(DEFAULT_ARUCO_DICT);
        cv::aruco::ArucoDetector detector(dictionary, detectorParams);

        detector.detectMarkers(image, markerCorners, markerIds, rejectedCandidates);
        std::vector<cv::Point2f> real_points = {
            {ARUCO_POS_X + ARUCO_SIZE,  ARUCO_POS_Y},
            {ARUCO_POS_X + ARUCO_SIZE,  ARUCO_POS_Y + ARUCO_SIZE},
            {ARUCO_POS_X,               ARUCO_POS_Y + ARUCO_SIZE},
            {ARUCO_POS_X,               ARUCO_POS_Y},
        };

        Mat perspective_tf = cv::getPerspectiveTransform(markerCorners[0], real_points);
        return perspective_tf;
}

Mat get_birdview_from_aruco(Mat image) {
    Mat perspective_tf = get_perspective_tf_from_aruco(image);
    Mat warped;
    cv::warpPerspective(image, warped, perspective_tf, {1500, 1500});
    return warped;
}


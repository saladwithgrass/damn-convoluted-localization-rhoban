#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <getopt.h>
#include "prerequisites/prerequisites.hpp"
#include "white_lines.cpp"

using cv::Mat;

void tune_green_filter(Mat image) {
    Mat green;
    const char* winname = "tuning green";
    cv::namedWindow(winname);

    int lo_hue = 0;
    int hi_hue = 255;
    int lo_sat = 0;
    int hi_sat = 255;
    int lo_val = 0;
    int hi_val = 255;

    cv::createTrackbar("hl", winname, &lo_hue, 255);
    cv::createTrackbar("sl", winname, &lo_sat, 255);
    cv::createTrackbar("vl", winname, &lo_val, 255);

    cv::createTrackbar("hh", winname, &hi_hue, 255);
    cv::createTrackbar("sh", winname, &hi_sat, 255);
    cv::createTrackbar("vh", winname, &hi_val, 255);
    char key = 0;
    Mat clipped;
    while (key != 'q') {
        clipped = green_filter_HSV(image, lo_hue, hi_hue, lo_sat, hi_sat);
        cv::imshow(winname, clipped);
        key = cv::waitKey(1);
    }
}

int main (int argc, char *argv[]) {
    int opt;
    char* image_filename = nullptr;
    while ( (opt = getopt(argc, argv, "i:")) != -1) {
        switch (opt) {
            case 'i':
                image_filename = optarg;
                break;
            case '?':
                if (optopt == 'i') {
                    std::cerr << "Option -" << static_cast<char>(optopt) << " requires an argument\n";
                } else {
                    std::cerr << "Unknown option: " << static_cast<char>(optopt) << "\n";
                }
                return 1;
            default:
                return 1;
        }
    }
    std::cout << "processing image" << image_filename << '\n';
    
    Mat image = cv::imread(image_filename);
    image = get_birdview_from_aruco(image);

    Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
    
    Mat integralY;
    Mat YCbCr;
    cv::cvtColor(image, YCbCr, cv::COLOR_BGR2YUV);
    std::vector<Mat> YCbCr_channels(3);
    cv::split(YCbCr, YCbCr_channels);
    integralY = YCbCr_channels[2];
    cv::integral(image, integralY);

    Mat green_filter;
    green_filter = green_filter_HSV(image);
    
    // cv::imshow("huh", green_filter);
    // cv::waitKey();
    // cv::destroyAllWindows();

    Mat bird_view = get_birdview_from_aruco(image);
    cv::imshow("huh", green_filter);
    // cv::waitKey();
    // cv::destroyAllWindows();
    std::cout << "begin segmentation\n";
    segment_white_lines(gray, integralY, green_filter);
    while (cv::waitKey() != 'q') { }
    cv::destroyAllWindows();
    return 0;
}

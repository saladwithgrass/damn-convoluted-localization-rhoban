#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <getopt.h>
#include <opencv2/core/hal/interface.h>
#include "prerequisites/prerequisites.hpp"
#include "white_lines.cpp"



using cv::Mat;

const std::vector<double> distortion_coeffs = {
    -0.2742843565205694, 0.015610089528832072, -6.426914987756399e-05, 0.00045607664773090356
};

double intr_array[] = {
    1281.3175182001776, 0, 980.2324643557556, 
    0, 1280.0658685934382,  498.43352671740007,
    0, 0, 1
};

const Mat intrinsics = Mat(3, 3, CV_64F, intr_array);

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
    
    Mat source = cv::imread(image_filename);
    Mat image;
    cv::undistort(source, image, intrinsics, distortion_coeffs);
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

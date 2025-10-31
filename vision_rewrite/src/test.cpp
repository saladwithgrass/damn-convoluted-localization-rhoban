#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <getopt.h>

int main (int argc, char *argv[]) {
    int opt;
    char* image_filename = nullptr;
    while ( (opt = getopt(argc, argv, "i:")) != -1) {
        switch (opt) {
            case 'i':
                image_filename = optarg;
                break;
            case '?':
                if (optopt == 'p') {
                    std::cerr << "Option -" << static_cast<char>(optopt) << " requires an argument\n";
                } else {
                    std::cerr << "Unknown option: " << static_cast<char>(optopt) << "\n";
                }
                return 1;
            default:
                return 1;
        }
    }
    
    cv::Mat image = cv::imread(image_filename);
    cv::Mat integral = c
    cv::imshow("huh", image);
    cv::waitKey();

    return 0;
}

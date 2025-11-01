#include <opencv2/opencv.hpp>

typedef struct CornerCandidate_struct 
{
    cv::Point2f ufa;
    cv::Point2f vfa;
    cv::Point2f ufb;
    cv::Point2f vfb;
    cv::Point2f corner;
    double score;
    bool valid;
} CornerCandidate;

#ifndef CAMERA_STATE_H_
#define CAMERA_STATE_H_

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Eigen>

class CameraState {
    public:
        Eigen::Affine3d worldToSelf;
        Eigen::Affine3d selfToWorld;
        cv::Point2f getPosInSelf(const cv::Point2f& pos_in_origin) const;
        cv::Point2f worldPosFromBirdviewImg(double imgX, double imgY);
        cv::Point2f robotPosFromBirdviewImg(double imgX, double imgY);  
        cv::Point2f getUnionSquareCenterOnBirdviewPix() const;
        double getYaw();
        int lineInfoFromPixel(
                const cv::Point2f &pos,
                float *dx, float *dy,
                int *px0, int *py0,
                int *px1, int *py1,
                int *px2, int *py2,
                int *px3, int *py3,
                double angularPitchError = -1.0
        );

    private:
        double birdviewPixelsInOneMeter;
        cv::Point2f unionSquareCenterOnBirdviewPix; //This union square will be projected at this coords in birdview image 
        cv::Size birdviewImageSize;

        Eigen::Vector3d unionSquareCenterForBirdviewMeters; //Center of union square should be in front of the camera far enough to be in frame
};

#endif // !CAMERA_STATE_H_

'''
Modded version of https://github.com/bvnayak/stereo_calibration/blob/master/camera_calibrate.py
'''
#stereoRectify  seems to work better in fresh opencv4.3.0 according to https://github.com/opencv/opencv/pull/10793/commits/b085158d5999fdced91094405f8753e9275092f6
#So let's use strictly this version 
import sys
sys.path.insert(0,'/home/egor/src/opencv430/opencv/build/lib/python3')
import cv2

import numpy as np
import glob
import argparse

sample_left = None
sample_right = None

'''
grid_size_x = 9
grid_size_y = 7
square_size_m = 0.02
'''
grid_size_x = 9
grid_size_y = 8
square_size_m = 0.08

# For now good stereo calibration can be done only using simple non-rational model, otherwise valid_pixel_roi is terrible
# So use USE_RATIONAL_MODEL=1 only to get "Kl_full" and "Dl" for classic rhoban pipeline (put results to calibration_wideangle_full.json) 
USE_RATIONAL_MODEL = 0

print (cv2.__version__)

np.set_printoptions(formatter={'float': '{: 0.9f},'.format}) #will output numbers as "1.000000000" with comma after each


class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria_corners = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((grid_size_x*grid_size_y, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:grid_size_x, 0:grid_size_y].T.reshape(-1, 2)
        self.objp *= square_size_m

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        self.read_images(self.cal_path)




    def read_images(self, cal_path):

        images_right = glob.glob(cal_path + '/right*.png')
        images_left = glob.glob(cal_path + '/left*.png')
        images_left.sort()
        images_right.sort()
        #print(images_left)

        for i, fname in enumerate(images_right):
            print ("Parsing image pair:" + repr(images_left[i]) + repr(images_right[i]) )

            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i]) 

            if i==41:
                sample_left = img_l
                sample_right = img_r

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (grid_size_x, grid_size_y), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (grid_size_x, grid_size_y), None)

            # If found, add object points, image points (after refining them)
            if (ret_l is True) and (ret_r is True):
                self.objpoints.append(self.objp)

                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria_corners)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                
                cv2.drawChessboardCorners(img_l, (grid_size_x, grid_size_y), corners_l, ret_l)
                #cv2.imshow(images_left[i], img_l)
                cv2.imshow("left", img_l)
                cv2.waitKey(5)

                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria_corners)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                cv2.drawChessboardCorners(img_r, (grid_size_x, grid_size_y), corners_r, ret_r)
                #cv2.imshow(images_right[i], img_r)
                cv2.imshow("right", img_r)
                cv2.waitKey(5)
            img_shape = gray_l.shape[::-1]

        cv2.destroyAllWindows()

        flags = 0
        if USE_RATIONAL_MODEL:
            flags |= cv2.CALIB_RATIONAL_MODEL

        print ("Calibrating single left cam...")
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None, flags=flags, criteria=self.criteria_cal)

        rectified_left = self.rectifyImage(sample_left, self.M1, self.d1, None, self.M1)
         
        #cv2.imshow("left_raw", sample_left)
        #cv2.imshow("left_rectified_single", rectified_left)

        print ("Calibrating single right cam...")
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None, flags=flags, criteria=self.criteria_cal)

        rectified_right = self.rectifyImage(sample_right, self.M2, self.d2, None, self.M2)

        #cv2.imshow("right_raw", sample_right)
        #cv2.imshow("right_rectified_single", rectified_right)
        visUR = np.concatenate((sample_left, sample_right), axis=1)
        visR = np.concatenate((rectified_left, rectified_right), axis=1)
        vis = np.concatenate((cv2.pyrDown(visUR), cv2.pyrDown(visR)), axis=0)
        cv2.imshow("single cam calibration", vis)
        cv2.waitKey(100)

        self.camera_model = self.stereo_calibrate(img_shape, sample_left, sample_right)

    def stereo_calibrate(self, dims, sample_left_, sample_right_):

        print ("Performing stereo calibration using existing intrinsics...")

        R = np.identity(3)

        T = np.zeros((3,1))
        T[0,0] = -0.06 # Use approximate camera-to-camera distance as a guess of T for calibration with CALIB_USE_EXTRINSIC_GUESS

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        if USE_RATIONAL_MODEL:
            flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5
        flags |= cv2.CALIB_USE_EXTRINSIC_GUESS

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        ret, M1, d1, M2, d2, R, T, E, F, perViewErrors = cv2.stereoCalibrateExtended(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims, R, T,
            criteria=stereocalib_criteria, flags=flags)
        print ("...done, perViewErrors=")
        print (perViewErrors)

        self.printMat('Kl_full', M1*2.0)
        self.printMat('Kl', M1)
        self.printMat('Kr', M2)
        self.printMat('Dl', d1)
        self.printMat('Dr', d2)

        flags_sr = 0
        flags_sr |= cv2.CALIB_ZERO_DISPARITY
        alpha = -1.0
        #alpha = 0.3
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(M1, d1, M2, d2, dims, R, T, flags = flags_sr, alpha=alpha)

        self.printMat('Rl', R1)
        self.printMat('Rr', R2)
        self.printMat('Pl', P1)
        self.printMat('Pr', P2)

        self.printMat('R', R)
        self.printMat('T', T)        


        rectified_left_stereo = self.rectifyImage(sample_left_, self.M1, self.d1, R1, P1)
        rectified_right_stereo = self.rectifyImage(sample_right_, self.M2, self.d2, R2, P2)

        vis = np.concatenate((rectified_left_stereo, rectified_right_stereo), axis=1)

        cv2.imshow("rectified_stereo", vis)

        cv2.waitKey(0)


        print('')

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])

        cv2.destroyAllWindows()
        return camera_model

    def rectifyImage(self, raw, K, D, R, P):

        imsize = (raw.shape[1], raw.shape[0])

        mapx, mapy = cv2.initUndistortRectifyMap(K, D, R, P, imsize, cv2.CV_32FC1)

        rectified = cv2.remap(raw, mapx, mapy, cv2.INTER_LINEAR)  
        return rectified

    def printMat(self, name, data):
        print('"xx" :'.replace('xx', name) + ' '.join(map(str, data)).replace('] [', '').replace(',]', ']') )


                  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    cal_data = StereoCalibration(args.filepath)

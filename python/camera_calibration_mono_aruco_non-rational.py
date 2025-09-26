"""
This code assumes that images used for calibration are of the same arUco marker board provided with code
Based on initial script by abhishek098 (https://github.com/abhishek098/camera_calibration/blob/master/camera_calibration.py)
Modded by Sol for Starkitrobots rhoban's wide angle setup

"""

import cv2
from cv2 import aruco
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

# path = "simpleCameraCalibrationNarrowQuarter"
path = "simpleCameraCalibrationWideFull"
# path = "simpleCameraCalibrationWideQuarter"
doPyrDownOnInputImages = False

# root directory of repo for relative path specification.
root = Path(__file__).parent.absolute()

# Set this flsg True for calibrating camera and False for validating results real time
calibrate_camera = True

# Set path to the images
#calib_imgs_path = root.joinpath("aruco_data")
calib_imgs_path = root

# For validating results, show aruco board to camera.
aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )

#Provide length of the marker's side
markerLength = 3.75  # Here, measurement unit is centimetre.

# Provide separation between markers
markerSeparation = 0.5   # Here, measurement unit is centimetre.

# create arUco board
board = aruco.GridBoard_create(4, 5, markerLength, markerSeparation, aruco_dict)

'''uncomment following block to draw and show the board'''
#img = board.draw((864,1080))
#cv2.imshow("aruco", img)

arucoParams = aruco.DetectorParameters_create()


if calibrate_camera == True:
    img_list = []
    calib_fnms = calib_imgs_path.glob(path + '/*.jpg')
    print('Using ...', end='')
    for idx, fn in enumerate(calib_fnms):
        print(idx, '', end='')
        img = cv2.imread( str(root.joinpath(fn) ))
        if doPyrDownOnInputImages: img = cv2.pyrDown(img)
        img_list.append( img )
        h, w, c = img.shape
    print('Calibration images')

    counter, corners_list, id_list = [], [], []
    first = True
    for im in tqdm(img_list):
        img_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        if first == True:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list,ids))
        counter.append(len(ids))
    print('Found {} unique markers'.format(np.unique(ids)))

    counter = np.array(counter)
    print ("Calibrating camera .... Please wait...")
    #mat = np.zeros((3,3), float)
    #_flags = cv2.CALIB_RATIONAL_MODEL + 0*cv2.CALIB_FIX_K3 + 0*cv2.CALIB_FIX_PRINCIPAL_POINT  # perfect undistortion but gives 8 distortion coeffs
    _flags = cv2.CALIB_FIX_K3
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape, cameraMatrix=None, distCoeffs=None, flags=_flags)

    print("Camera matrix: \n", mtx)
    print("Distortion coefficients: \n", dist)
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist(), 'width': img_list[0].shape[1], 'height': img_list[0].shape[0]}
    with open(path + "/calibration.yaml", "w") as f:
        yaml.dump(data, f)


print("Rectifying image")
#else:
if calibrate_camera == True:
    img_list = []
    calib_fnms = calib_imgs_path.glob(path + '/*.jpg')
    for idx, fn in enumerate(calib_fnms):
        print(idx, '', end='')
        img = cv2.imread( str(root.joinpath(fn) ))
        if doPyrDownOnInputImages: img = cv2.pyrDown(img)
        img_list.append( img )
    #camera = cv2.VideoCapture(0)
    #ret, img = camera.read()
    img =  img_list[0]

    with open(path + '/calibration.yaml') as f:
        loadeddict = yaml.load(f)
    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    dist = np.array(dist)

    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    h,  w = img_gray.shape[:2]
    alpha = 1.0
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),alpha,(w,h))

    # [Sol]
    # caemra_matrix_optimal is a new camera matix for wide angle imageunwarp with better frame coverance
    # But it shoud be used as an optional FIFTH ARGUMENT in cv2.undistrort,
    # not as an 'main' camera matrix in first argument. Attemp to integrate this additional camera matrix
    # to rhoban's stack seems not costs the effort and was abandoned by now
    # so don't even save this new camera matrix

    #data = {'camera_matrix_optimal': np.asarray(newcameramtx).tolist(), 'dist_coeff_optimal': np.asarray(dist).tolist()}
    #with open("calibration_optimal.yaml", "w") as f:
    #    yaml.dump(data, f)
    #print(roi)
    #newcameramtx = mtx

    pose_r, pose_t = [], []
    for img in img_list:
        img_aruco = img
        im_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        h,  w = im_gray.shape[:2]
        #dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # see comment above about caemra_matrix_optimal
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(im_gray, aruco_dict, parameters=arucoParams)

        #cv2.imshow("original", img_gray)
        if corners == None:
            print ("pass")
        else:
            #print (corners[0])
            pts = corners[0]
            #out = cv2.undistortPoints(pts.reshape(-1,1,2).astype(np.float32), newcameramtx, dist)
            out = cv2.undistortPoints(pts.reshape(-1,1,2).astype(np.float32), mtx, dist)
            print(out)
            for pt in out:
                pt_ = pt[0]
                #fx = newcameramtx[0][0]
                #fy = newcameramtx[1][1]
                #cx = newcameramtx[0][2]
                #cy = newcameramtx[1][2]
                fx = mtx[0][0]
                fy = mtx[1][1]
                cx = mtx[0][2]
                cy = mtx[1][2]
                print( (fx,fy,cx,cy) )
                center = (int(pt_[0]*fx + cx), int(pt_[1]*fy + cy))
                cv2.circle(dst, center, 10, (255,0,255), -1)
            #ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, dist) # For a board
            #print ("Rotation ", rvec, "Translation", tvec)
            #if ret != 0:
            #    img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
            #    img_aruco = aruco.drawAxis(img_aruco, newcameramtx, dist, rvec, tvec, 10)    # axis length 100 can be changed according to your requirement

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break;
        #cv2.imshow("World co-ordinate frame axes", img_aruco)
        src_small = cv2.pyrDown(img)
        cv2.imshow("Distorted", src_small)
        dst_small = cv2.pyrDown(dst)
        cv2.imshow("Undistorted", dst_small)

cv2.destroyAllWindows()

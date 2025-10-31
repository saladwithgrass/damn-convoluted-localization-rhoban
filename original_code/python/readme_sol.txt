At the moment (2020/2021) we are using two different (!) camera models in KidSize project.

1.For stereo vision/etc we are using standart pinhole (plumb-bob, etc) camera model
with classic NON-RATIONAL distortion (i.e. only 5 coeffs are needed - [k1,k2,t1,t2,k3].
So stereo rig should be calibrated with large enough printed chessboard pattern.

Calibraton images for stereo rig can be obtained setting "doCalibration=1" in stereoImgProc, output images will be in rhobot's /tmp

Use script "camera_calibration_stereo_chess_non-rational.py"
Calibration results should be placed in "calibration_stereo.json" as is (btw add commas at all lines excepl the last one)

2. For all other classic rhoban vision pipeline we are using RATIONAL distortion model to better take
into account points close to lens FOV limit (ball detection, TaggedImg drawing, etc).

To calibrate using same images taken from stereoImgProc - use "camera_calibration_mono_chess_rational_from_stereo_images.py",
it will output proper part of "calibration_wideangle_full.json".
Calibration results will contains 8 distortion coeffs.

Or calibrate using small aruco buard and "camera_calibration_mono_aruco_rational.py".


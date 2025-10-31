# Questions

## We apply green filter to the image and segment white lines.

## How do we get exact straight lines from the segmented image?
### Get initial images
To procees an image, we get it in the grayscale colorspace, image with
integrated luminosity and green color density image. Green color density is
extracted via integral image of green.

### Make green brighter
Grayscale image is converted to bgr. Every color except green is made half as
bright:
```
        im.at<Vec3b>(y,x) = Vec3b(clr(0)/2, clr(1), clr(2)/2);
```

### Approximate line widths
Three points on the field are selected with hardcode.
Then, line width in these points is approximated with
`CameraState::lineInfoFromPixel`.

#### `lineInfoFromPixel`
First, we get a raycast from the center of the image as a Ray object.:
```
  Ray cameraCentralRay = getRayInWorldFromPixel(cv::Point2f(getImgSize().width / 2, 0));
```
Then, we set forward and left directions in 2d(z coordinate is zero).
forward direction is a normalized raycast from the center of the image. Left
direction is perpendicular to the forward direction.
```
  Eigen::Vector3d forwardDir = Eigen::Vector3d(cameraCentralRay.dir(0), cameraCentralRay.dir(1), 0.0).normalized();
  Eigen::Vector3d leftDir = Eigen::Vector3d(cameraCentralRay.dir(1), -cameraCentralRay.dir(0), 0.0).normalized();
```
\
Third, we get a raycast from target pixel and make sure that it is not looking
above horizon. Let's call this ray `viewRay`.
```
  Ray viewRay = getRayInWorldFromPixel(pos);
```

Then, intersection of viewRay and ground plane is calculated.
Point of intersection then is reduced to 2 dimensions by omitting z-value.

After calculating the point of intersection, two shift vectors are defined: `shiftxr` and `shiftyr`. 
`shiftxr` is left direction vector multiplied by 0.005/2.
`shiftyr` is forward direction vector multiplied by 0.005/2.
We have a point in 2d space. We get 4 points around it. Each of these points
is 0.005/2 meters away from the central point. These points are shifted by x
and by y.
We get these points' pixel coordinates:
```
    pim0sx = imgXYFromWorldPosition(pof - shiftxr);  // Get pixel coords of shifted point on field
    pim1sx = imgXYFromWorldPosition(pof + shiftxr);  // Get pixel coords of shifted point on field
    pim0sy = imgXYFromWorldPosition(pof - shiftyr);  // Get pixel coords of shifted point on field
    pim1sy = imgXYFromWorldPosition(pof + shiftyr);  // Get pixel coords of shifted point on field
```
Let's say that the center point is (0, 0) and `delta = 0.005/2`. The 4 points surrounding it are:
* (-delta, 0)
* (delta, 0)
* (0, delta)
* (0, -delta)

Also, we have calculated their pixel coordinates.
Now we know, how `delta`(in meters) converts to pixels in this point of the
image. This will allow us to calculate line width.

### Plane Fitting
Now, we have three 


## How do we extract features from lines?

## How do we connect two features between themselves?

## How to localize based on features?
It seems that the answer may lien in RobotFilter.cpp.

The robot uses a particle filter for localisation.
- Vision/Localisation/Field/ contains all the code relevant to the particle filter
- Vision/Binding/LocalisationBinding contains the interface of the localisation
  module with other modules


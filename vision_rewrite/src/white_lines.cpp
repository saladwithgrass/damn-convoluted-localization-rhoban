int CameraState::lineInfoFromPixel(
        const cv::Point2f& pos, 
        float* dx, float* dy, 
        int* px0, int* py0, 
        int* px1, int* py1,
        int* px2, int* py2, 
        int* px3, int* py3, 
        double angularPitchError) {
  // Ray cameraCentralRay = getRayInWorldFromPixel(cv::Point2f(getImgSize().width/2, getImgSize().height/2));

  Ray cameraCentralRay = 
      getRayInWorldFromPixel(cv::Point2f(getImgSize().width / 2, 0));
  // Ray cameraCentralRay = getRayInWorldFromPixel(pos);
  // Ray consist of source and dir, both Eigen3D

  Eigen::Vector3d forwardDir = Eigen::Vector3d(cameraCentralRay.dir(0), cameraCentralRay.dir(1), 0.0).normalized();
  Eigen::Vector3d leftDir = Eigen::Vector3d(cameraCentralRay.dir(1), -cameraCentralRay.dir(0), 0.0).normalized();
  // Eigen::Vector3d leftDir = Eigen::Vector3d(0.0, 0.0, 0.0);
  // Eigen::Vector3d leftDir = viewRay.dir.cross(groundDir).normalized();

  Ray viewRay = getRayInWorldFromPixel(pos);
  if (viewRay.dir.z() >= 0)  // Point above horison (z componnet of direction >0)
  {
    return -1;
  }
  Plane groundPlane(Eigen::Vector3d(0, 0, 1), 0.0);

  if (!isIntersectionPoint(viewRay, groundPlane)) {
    return -1;
  }

  Eigen::Vector3d posInWorld = getIntersection(viewRay, groundPlane);
  cv::Point2f pof = cv::Point2f(posInWorld(0), posInWorld(1));

  cv::Point2f shiftxr = cv::Point2f(leftDir(0) * 0.05 / 2, leftDir(1) * 0.05 / 2);
  cv::Point2f shiftyr = cv::Point2f(forwardDir(0) * 0.05 / 2, forwardDir(1) * 0.05 / 2);

  /*
  //Calculating vector to shift pixel to half of line width in "x" (from left to right) direction in image taking into
  account camera Yaw cv::Point2f shiftx = cv::Point2f(0.0, 0.05/2.0); //Shift to the left cv::Point2f shiftxr =
  cv::Point2f(shiftx.x * cos(getYaw()) - shiftx.y * sin(getYaw()), shiftx.x * sin(getYaw()) + shiftx.y * cos(getYaw())
  );

  //Calculating vector to shift pixel to half of line width in "y" (from bottom to top) direction in image taking into
  account camera Yaw
  //cv::Point2f shifty = cv::Point2f(0.05/2.0, 0.00); //Shift forward
  cv::Point2f shifty = cv::Point2f(0.0, 0.00); //Shift forward
  cv::Point2f shiftyr = cv::Point2f(shifty.x * cos(getYaw()) - shifty.y * sin(getYaw()),
                                   shifty.x * sin(getYaw()) + shifty.y * cos(getYaw()) );

  //cv::Point2f pof = robotPosFromImg(pos.x, pos.y); //Get point on field from pixel coords
  cv::Point2f pof = worldPosFromImg(pos.x, pos.y); //Get point on field from pixel coords, 2019 version
  */
  cv::Point2f pim0sx, pim1sx, pim0sy, pim1sy;

  try {
    pim0sx = imgXYFromWorldPosition(pof - shiftxr);  // Get pixel coords of shifted point on field
    pim1sx = imgXYFromWorldPosition(pof + shiftxr);  // Get pixel coords of shifted point on field
    pim0sy = imgXYFromWorldPosition(pof - shiftyr);  // Get pixel coords of shifted point on field
    pim1sy = imgXYFromWorldPosition(pof + shiftyr);  // Get pixel coords of shifted point on field
  } catch (const std::runtime_error& exc) {
    return -1;
  }

  /*
  //cv::Point2f pim0sx = imgXYFromRobotPosition2f(pof - shiftxr, width, height); //Get pixel coords of shifted point on
  field
  //cv::Point2f pim1sx = imgXYFromRobotPosition2f(pof + shiftxr, width, height); //Get pixel coords of shifted point on
  field
  //cv::Point2f pim0sy = imgXYFromRobotPosition2f(pof - shiftyr, width, height); //Get pixel coords of shifted point on
  field
  //cv::Point2f pim1sy = imgXYFromRobotPosition2f(pof + shiftyr, width, height); //Get pixel coords of shifted point on
  field
  */

  *px0 = pim0sx.x;
  *py0 = pim0sx.y;
  *px1 = pim1sx.x;
  *py1 = pim1sx.y;
  *px2 = pim0sy.x;
  *py2 = pim0sy.y;
  *px3 = pim1sy.x;
  *py3 = pim1sy.y;

  // *dx = fabs(pim0sx.x - pim1sx.x);
  // *dy = fabs(pim0sy.y - pim1sy.y);
  *dx = sqrt(pow(pim0sx.x - pim1sx.x, 2) + pow(pim0sx.y - pim1sx.y, 2));
  *dy = sqrt(pow(pim0sy.x - pim1sy.x, 2) + pow(pim0sy.y - pim1sy.y, 2));

  return 0;  // no errors
}

void WhiteLines::process() {

  //std::cout << "-------------- WHITE LINES PROCRESS() ENTER ------------" << std::endl;
  Vision::Utils::CameraState *cs = &getCS();

  // should get 
  cv::Mat source = (getDependency(sourceName).getImg())->clone();

  cv::Mat integralY = (getDependency(integralYName).getImg())->clone();

  cv::Mat green = (getDependency(greenName).getImg())->clone();  

  int row_nb = source.rows;
  int col_nb = source.cols;

  //std::cout << "-------------- WHITE LINES PROCRESS() DEPENDENCY GET OK ------------" << std::endl;
  //Defining verbose image
  cv::Mat im(row_nb, col_nb, CV_8UC3, cv::Scalar(0,0,0));
  
  //Copy source gray image to verbose output
  cv::cvtColor(source, im, CV_GRAY2BGR);
  
  //Mark green filter output on verbose image for debug
  for(int y=0; y < im.rows; y++) {
    for(int x=0; x < im.cols; x++) {
      if(green.at<uchar>(y,x) > 0) {
        Vec3b clr = im.at<Vec3b>(y,x);
        im.at<Vec3b>(y,x) = Vec3b(clr(0)/2, clr(1), clr(2)/2);
      }
    }
  }



  //To debug input image:
  //source.copyTo(img());
  //return;

  //Definig clipping image
  cv::Mat fieldOnly = cv::Mat(row_nb, col_nb, CV_8UC1);
  fieldOnly.setTo(255);

  //Clipping by horizon

  //DON'T USE yNoRobot inetgral image as in default 2019 env here, it will contain holes whom will ruin all processing!!!  
  Mat isum = integralY; //Pick ready integral image from rhoban's pipeline

  Mat iresx = Mat(im.rows, im.cols, CV_8UC1);
  Mat iresy = Mat(im.rows, im.cols, CV_8UC1);
  Mat iresrgb = Mat(im.rows, im.cols, CV_8UC3);
  iresx.setTo(0);
  iresy.setTo(0);
  iresrgb.setTo(0);
  
  //At first we need to get proper line size in image x and y direction in any point of an image.
  //To do it quick we use plane-based interpolation estimating a plane A*x+B*y+C*z+D=0 coeffs where x and y are pisel coords, and z is interpolated line width at this point
  //TODO: seems that this plane fitting method is alternative reimplementation of BallRadiusProvider by rhoban authors :) Check which approach is better
  double planex_a, planex_b, planex_c, planex_d; //Plane coeffs for line width in x direction
  double planey_a, planey_b, planey_c, planey_d; //Plane coeffs for line width in y direction

  //To fit the plane, we choose 3 points in hardcoded coords. Maybe do some smarter estimation with no hardcode?
  cv::Point2f ls_pim0 = cv::Point((float)source.cols*0.2, (float)source.rows*0.8);
  cv::Point2f ls_pim1 = cv::Point((float)source.cols*0.8, (float)source.rows*0.8);
  cv::Point2f ls_pim2 = cv::Point((float)source.cols*0.5, (float)source.rows*0.4);

  int px0,py0, px1,py1, px2,py2, px3,py3;
  float dx0, dy0, dx1, dy1, dx2, dy2;
  //getting line width components for this 3 points
  int res1 = cs->lineInfoFromPixel(ls_pim0, &dx0, &dy0, &px0, &py0, &px1, &py1, &px2, &py2, &px3, &py3);
  int res2 = cs->lineInfoFromPixel(ls_pim1, &dx1, &dy1, &px0, &py0, &px1, &py1, &px2, &py2, &px3, &py3);
  int res3 = cs->lineInfoFromPixel(ls_pim2, &dx2, &dy2, &px0, &py0, &px1, &py1, &px2, &py2, &px3, &py3);
  
  if(res1 || res2 || res3) {
    //In case of a point above horizon
    //std::cout << "-------------- WHITE LINES PROCRESS() POINT ABOVE HORIZON! ------------" << std::endl;
    
    return;
  }

  //std::cout << "dx0="<< dx0 <<", dy0=" << dy0 << std::endl;
  //std::cout << "dx1="<< dx1 <<", dy1=" << dy1 << std::endl;
  //std::cout << "dx2="<< dx2 <<", dy2=" << dy2 << std::endl;


  //Estimating plane coeffs
  get_plane_equation(
          ls_pim0.x, // x1
          ls_pim0.y, // y1
          dx0+3,     // z1
          ls_pim1.x, // x2
          ls_pim1.y, // y2
          dx1+3,     // z2
          ls_pim2.x, // x3
          ls_pim2.y, // y3
          dx2+3,     // z3
          &planex_a,
          &planex_b,
          &planex_c,
          &planex_d
  );
  get_plane_equation(
          ls_pim0.x,
          ls_pim0.y,
          dy0+3,
           ls_pim1.x,
          ls_pim1.y,
          dy1+3,
           ls_pim2.x,
          ls_pim2.y,
          dy2+3,
          &planey_a,
          &planey_b,
          &planey_c,
          &planey_d
  );
 
 
  //std::cout << "-------------- WHITE LINES PROCRESS() PLANE EQUATION OK ------------" << std::endl;
  //Widnow size orthogonal to traverse direction

  Benchmark::open("Integral image traversing");
  int win_height=10; 

  //Divider to check only Nth scanline. Greatly improves speed with almost no quality degradation. 4 seems to be enoughth for everybody (c)
  int scanline_divider = 4;

  
  //Now we can traverse image in x and y direction and do some integral image processing using window with 3 parts - left part, middle part, right part.
  //Each part have width equal to assumed line width in this location

  //x traverse/scalnine
  for(int yn=0; yn < (im.rows-win_height)/scanline_divider; yn++) {
      int y = yn*scanline_divider;
        for(int x=0; x < im.cols-1; x++) {
          int win_line_width = -(planex_a*x + planex_b*y + planex_d)/planex_c;

          if(win_line_width<=0) continue;
          if(win_line_width<3) win_line_width = 3;

          if (x>=im.cols-win_line_width*3-1) break; //not to hit the border

          int sleft   = getRegionSum(isum, x,                  y, win_line_width, win_height);
          int smiddle = getRegionSum(isum, x+win_line_width,   y, win_line_width, win_height);
          int sright  = getRegionSum(isum, x+win_line_width*2, y, win_line_width, win_height);

          int diff = (smiddle-sleft)*(smiddle-sright)
              /(win_line_width*win_height)
              /(win_line_width*win_height)
              / 20;
          if((smiddle-sright)<0) diff=0;
          if((smiddle-sleft)<0) diff=0;
          if(diff<0) diff = 0;
          if(diff>255) diff = 255;
          iresx.at<uchar>(y+win_height/2,x+(win_line_width*3)/2) = diff;
          iresx.at<uchar>(y+win_height/2,x+(win_line_width*3)/2+1) = diff; //Dirty hack to remove empty pixels when linewidth changes a lot, TODO: refactor - start the window from meddle, not from left edge
      }
  }

  //y traverse/scanline
  for(int xn=0; xn < (im.cols-win_height)/scanline_divider; xn++) {
      int x = xn*scanline_divider;
        for(int y=0; y < im.rows-1; y++) {
          int win_line_width = -(planey_a*x + planey_b*y + planey_d)/planey_c;
          if(win_line_width<=0) continue;
          if(win_line_width<3) win_line_width = 3;
          if (y>=im.rows-win_line_width*3-1) break; //not to hit the border          
          int sleft   = getRegionSum(isum, x,y, win_height, win_line_width);
          int smiddle = getRegionSum(isum, x,y+win_line_width, win_height, win_line_width);
          int sright  = getRegionSum(isum, x,y+win_line_width*2, win_height, win_line_width);
          int diff = (smiddle-sleft)*(smiddle-sright)/(win_line_width*win_height)/(win_line_width*win_height)/20;
          if((smiddle-sright)<0) diff=0;
          if((smiddle-sleft)<0) diff=0;          
          if(diff<0) diff = 0;
          if(diff>255) diff = 255;
          iresy.at<uchar>(y+(win_line_width*3)/2,x+win_height/2) = diff;
          iresy.at<uchar>(y+(win_line_width*3)/2+1,x+win_height/2) = diff; //Dirty hack to remove empty pixels when linewidth changes a lot, TODO: refactor - start the window from meddle, not from left edge
      }
  }

  Benchmark::close("Integral image traversing");

  Benchmark::open("Non-maxima suppression");

  //Now let's do non-maxima suppression on traverse results - this will thin results to single pixel width in the middle of detected line (if line quality is not too bad)
  Mat nmsx = cv::Mat(row_nb, col_nb, CV_8UC1);
  Mat nmsy = cv::Mat(row_nb, col_nb, CV_8UC1);

  //Suppress local maximas in 30 pixel window with intensity thresold
  int line_intencity_treshold = 5; //with treshold=3 even wery subtile lines are detected, but too many false positives (edges of a ball, etc)
  //int line_intencity_treshold = 100;

  //TODO: remove hardcode
  non_maxima_suppression(iresx, nmsx, 30, 1, line_intencity_treshold);
  non_maxima_suppression(iresy, nmsy, 1, 30, line_intencity_treshold);

  //Resulting mat for both x and y results in single channel
  Mat nms = cv::Mat(row_nb, col_nb, CV_8UC1);
  nms.setTo(0);

  //Checking for proper green color at left and right sides of a line
  for(int y=0; y < im.rows-win_height; y++) {
    for(int x=0; x < im.cols; x++) {
      if(nmsx.at<uchar>(y,x)>0) {        
        int val = 255;
        im.at<Vec3b>(y,x) = Vec3b(255, 0, 0);
        int win_line_width = -(planex_a*x + planex_b*y + planex_d)/planex_c;
        if(win_line_width<=0) continue;
        if(win_line_width<3) win_line_width = 3;
        if (x>=im.cols-win_line_width*3-1) break; //not to hit the border
        int gleft = countNonZero(green(Rect(x,y, win_line_width,win_height)));
        int gright = countNonZero(green(Rect(x+win_line_width*2,y, win_line_width,win_height)));
        if(gleft < 0.05*(float)win_line_width*(float)win_height) val = 0;  //5% green coverage, TODO: remove hardcode 
        if(gright < 0.05*(float)win_line_width*(float)win_height) val = 0;  //5% green coverage, TODO: remove hardcode         
        if(val>0) nms.at<uchar>(y,x) = val;
      }
    }
  }
  for(int x=0; x < im.cols-win_height; x++) {
    for(int y=0; y < im.rows; y++) {
      if(nmsy.at<uchar>(y,x)>0) {
        im.at<Vec3b>(y,x) = Vec3b(255, 0, 0);
        int val = 255;
        int win_line_width = -(planey_a*x + planey_b*y + planey_d)/planey_c;
        if(win_line_width<=0) continue;
        if(win_line_width<3) win_line_width = 3;
        if (y>=im.rows-win_line_width*3-1) break; //not to hit the border 
        int gleft = countNonZero(green(Rect(x,y, win_height,win_line_width)));
        int gright = countNonZero(green(Rect(x,y+win_line_width*2, win_height,win_line_width)));       
        if(gleft < 0.05*(float)win_line_width*(float)win_height) val = 0;  //5% green coverage, TODO: remove hardcode
        if(gright < 0.05*(float)win_line_width*(float)win_height) val = 0;  //5% green coverage, TODO: remove hardcode        
        if(val>0) nms.at<uchar>(y,x) = val;
      }

    }
  }



  
  std::vector<cv::Point2f> circle_ransac_input;

  //For debug verbose
  int show_linedetector_heatmap = 0;
  if(show_linedetector_heatmap) {
    im.setTo(0);
    for(int y=0; y < iresrgb.rows; y++) {
      for(int x=0; x < iresrgb.cols; x++) {
        //im.at<Vec3b>(y,x) = Vec3b(source.at<uchar>(y,x), iresy.at<uchar>(y,x), iresx.at<uchar>(y,x));
        im.at<Vec3b>(y,x) = Vec3b(0, iresy.at<uchar>(y,x), iresx.at<uchar>(y,x));
        //im.at<Vec3b>(y,x) = Vec3b(nms.at<uchar>(y,x), nms.at<uchar>(y,x), nms.at<uchar>(y,x));
      }
    }      
    im.copyTo(img());
    return;
  } else {
    for(int y=0; y < iresrgb.rows; y++) {
      for(int x=0; x < iresrgb.cols; x++) {
        if(nms.at<uchar>(y,x)>0) {
          im.at<Vec3b>(y,x) = Vec3b(0, 255, 255);
          
          //Filling data vector for central circle RANSAC
          //cv::Point2f pc = cs->robotPosFromImg(x, y, source.cols, source.rows);
          cv::Point2f pc;
          try {
            pc = cs->robotPosFromImg(x, y);
          } catch (const std::runtime_error& exc) {
            continue;
          }            
          circle_ransac_input.push_back(pc);
        }
      }
    }      
  }

  Benchmark::close("Non-maxima suppression");
  

  //std::cout << "-------------- WHITE LINES PROCRESS() NMS OK ------------" << std::endl; 
  //Ransac for central circle
  /*
  if(circle_ransac_input.size()>3) {
    int ransac_circle_inliers = 0;
    cv::Point2f circle_center;
    float circle_radius;
    ransac_circle_inliers =  ransac_circle(circle_ransac_input, circle_center, circle_radius);
    if(ransac_circle_inliers>(float)180*0.25) {
      //good circle is a circle with percentage of inliers more than 50%, 
      //or more than 25% inliers and more than 50% of points not in FOV
      int points_in_fov = 0;
      std::vector<cv::Point> circle_verbose_points;
      for(float t = 0; t<2.0*M_PI; t+= 6.0*M_PI/180.0) //total 60 points per circle
       {
          float rX = circle_radius*cos(t) + circle_center.x;
          float rY = circle_radius*sin(t) + circle_center.y;
          //TODO: imgXYFromRobotPosition is very time consuming
          cv::Point cp = cs->imgXYFromRobotPosition(cv::Point2f(rX, rY), source.cols, source.rows);
          if( (cp.x<source.cols) && (cp.x>0) && (cp.y<source.rows) && (cp.y>0) ) {
            circle_verbose_points.push_back(cp);
            points_in_fov++;
          }
      }
      if( (ransac_circle_inliers>(float)180*0.5) || ( (ransac_circle_inliers>(float)180*0.25)&&(points_in_fov<(float)60*0.5) ) ) {
        std::cout << "Found circle: center=" << circle_center << ", radius=" << circle_radius << ", inliers=" << ransac_circle_inliers <<", points_in_fov=" << points_in_fov << std::endl;
        for(int i=0;i<circle_verbose_points.size();i++) {
          //circle(im, circle_verbose_points[i], 2, Scalar(255,0,255), -1);  
          //im.at<cv::Vec3b>(circle_verbose_points[i].y, circle_verbose_points[i].x) = cv::Vec3b(255,0,255);
          cv::line(im, circle_verbose_points[i], circle_verbose_points[(i+1)%circle_verbose_points.size()], cv::Scalar(255,0,255), 2, LINE_AA);
        }
        cv::Point cc = cs->imgXYFromRobotPosition(circle_center, source.cols, source.rows);
        circle(im, cc, 10, Scalar(255,0,255), 2);  
      }
    }
  }
  */

  Benchmark::open("Probabilistic Hough Transform");

  //Probabilistic Line Transform after NMS and green check
  dilate(nms, nms, Mat(), Point(-1, -1), 1, 1, 1);
  vector<Vec4i> linesP_all;
  int pointsThreshold = 100/scanline_divider;
  int lengthThreshold = 50;
  int gapThreshold = 15; //25 from 2018 version gives too musk false positives;
  HoughLinesP(nms, linesP_all, 2, 2*CV_PI/180, pointsThreshold, lengthThreshold, gapThreshold );

  //Extending lines length a little bit for more robust intersecion detection
  double extend_in_pix = 2.0;
  for( size_t i = 0; i < linesP_all.size(); i++ )
  {
      Vec4i l = linesP_all[i];
      double len = sqrt(pow(l[2]-l[0], 2) + pow(l[3]-l[1], 2));
      double ldx = (l[2]-l[0])/len;
      double ldy = (l[3]-l[1])/len;
      linesP_all[i][2] += ldx * extend_in_pix;
      linesP_all[i][0] -= ldx * extend_in_pix;
      linesP_all[i][3] += ldy * extend_in_pix;
      linesP_all[i][1] -= ldy * extend_in_pix;
  }

  Benchmark::close("Probabilistic Hough Transform");

  Benchmark::open("Merging segments");
  //Merging similar segments
  //TODO: Not effective and leave small lines as artifacts, need to be rewritten (!)
  bool changes_done_line = true;
  while(changes_done_line) {
    changes_done_line = false;
    for( size_t i = 0; i < linesP_all.size(); i++ ) {
      //for( size_t j = i+1; j < linesP_all.size(); j++ ) {
      for( size_t j = 0; j < linesP_all.size(); j++ ) {        
        if( (i!=j) 
         && (get_line_magnitude(linesP_all[i][0], linesP_all[i][1], linesP_all[i][2], linesP_all[i][3])>1.0) 
         && (get_line_magnitude(linesP_all[j][0], linesP_all[j][1], linesP_all[j][2], linesP_all[j][3])>1.0) ) {

          if( (get_segnent2segment_distance(linesP_all[i][0], linesP_all[i][1], linesP_all[i][2], linesP_all[i][3],
                                          linesP_all[j][0], linesP_all[j][1], linesP_all[j][2], linesP_all[j][3])<10.0)

            &&(fabs(get_line2line_angle(linesP_all[i][0], linesP_all[i][1], linesP_all[i][2], linesP_all[i][3],
                                  linesP_all[j][0], linesP_all[j][1], linesP_all[j][2], linesP_all[j][3]))*180.0/M_PI<5.0) ) {
            //This segments should be merged
            float mx1, my1, mx2, my2;
            merge_two_segments( linesP_all[i][0], linesP_all[i][1], linesP_all[i][2], linesP_all[i][3],
                                linesP_all[j][0], linesP_all[j][1], linesP_all[j][2], linesP_all[j][3],
                                &mx1, &my1, &mx2, &my2);
            linesP_all[j][0] = mx1;
            linesP_all[j][1] = my1;
            linesP_all[j][2] = mx2;
            linesP_all[j][3] = my2;
            linesP_all[i][0] = 0;
            linesP_all[i][1] = 0;
            linesP_all[i][2] = 0;
            linesP_all[i][3] = 0;
            changes_done_line = true;          
          }
        }
      }
    }
  }

  Benchmark::close("Merging segments");

  Benchmark::open("Drawing segments");

  vector<Vec4i> linesP;
  for( size_t i = 0; i < linesP_all.size(); i++ )
  {
    if(get_line_magnitude(linesP_all[i][0], linesP_all[i][1], linesP_all[i][2], linesP_all[i][3])>1.0) linesP.push_back(linesP_all[i]);  
  }
  //std::cout << "Segments merge results: before=" << linesP_all.size() <<", after=" << linesP.size() << std::endl;



  // Draw all found segments
  for( size_t i = 0; i < linesP.size(); i++ )
  {
      //Scalar color = Scalar(rand()%255,rand()%255,rand()%255 );
      Vec4i l = linesP[i];
      line( im, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,175,100), 1);
      //line( im, Point(l[0], l[1]), Point(l[2], l[3]), color, 1, LINE_AA);
  }
   
  Benchmark::close("Drawing segments");
  /*
  //Draw the plane fitting trapezoid
  cv::Point2f shift = cv::Point2f(0.05, 0);
  cv::Point2f shiftr = cv::Point2f(shift.x * cos(cs->getYaw()) - shift.y * sin(cs->getYaw()), 
                                   shift.x * sin(cs->getYaw()) + shift.y * cos(cs->getYaw()) );

  cv::Point pim0 = cv::Point((float)source.cols*0.2, (float)source.rows*0.8);
  cv::Point2f pof0 = cs->robotPosFromImg(pim0.x, pim0.y, source.cols, source.rows); //Get point on field from pixel coords
  cv::Point pim0s = cs->imgXYFromRobotPosition(pof0 + shiftr, source.cols, source.rows); //Get pixel ccords of shifted point on field

  cv::Point pim1 = cv::Point((float)source.cols*0.8, (float)source.rows*0.8);
  cv::Point2f pof1 = cs->robotPosFromImg(pim1.x, pim0.y, source.cols, source.rows); //Get point on field from pixel coords
  cv::Point pim1s = cs->imgXYFromRobotPosition(pof1 + shiftr, source.cols, source.rows); //Get pixel ccords of shifted point on field

  line(im, pim0, pim1, Scalar(255,0,0), 1, LINE_AA);
  line(im, pim0, pim0s, Scalar(255,0,0), 1, LINE_AA);
  line(im, pim0s, pim1s, Scalar(255,0,0), 1, LINE_AA);
  line(im, pim1, pim1s, Scalar(255,0,0), 1, LINE_AA);*/

  //iresrgb.copyTo(img());
  //im.copyTo(img());

  Benchmark::open("Checking lines for intersections");
	//Check for intersections and fill an array of corner candidates
  std::vector<CornerCandiate> cornerCandidates;  
  float maxDist = std::max(Constants::field.field_length + Constants::field.border_strip_width_x*2, Constants::field.field_width + Constants::field.border_strip_width_x*2);
  for( size_t i = 0; i < linesP.size(); i++ )
  {
		for( size_t j = i+1; j < linesP.size(); j++ )
	  {
		  float i_x, i_y;
			Vec4i la = linesP[i];
			Vec4i lb = linesP[j];
		     char intersect = get_line_intersection(la[0], la[1], la[2], la[3], 
	  		 				                                lb[0], lb[1], lb[2], lb[3],
							                                  &i_x, &i_y);
			if(intersect) {
        cv::Point2f ps0, ps1, ps2, ps3, pi;
        try
        {
          ps0 = cs->robotPosFromImg(la[0], la[1]);
          ps1 = cs->robotPosFromImg(la[2], la[3]);
          ps2 = cs->robotPosFromImg(lb[0], lb[1]);
          ps3 = cs->robotPosFromImg(lb[2], lb[3]);
          pi = cs->robotPosFromImg(i_x, i_y);
        }
          catch (const std::runtime_error& exc)
        {
          continue;
        }
        float length_a = my_norm(ps0,ps1);
        float length_b = my_norm(ps2,ps3);
				if( (ps0.x < maxDist)&&(ps0.y < maxDist)&&(ps1.x < maxDist)&&(ps1.y < maxDist)&& //intersection point is farther than field size
				    (ps2.x < maxDist)&&(ps2.y < maxDist)&&(ps3.x < maxDist)&&(ps3.y < maxDist)&& //intersection point is farther than field size
				    ( sqrt(pow(pi.x,2)+ pow(pi.y,2)) >0.2 )&&  //Also skip here 0.2 meters radius zone around robot to ignore handling bar, etc
            (length_a>0.3 && length_b>0.3) ) //filter out penalty mark
        {
					float thetaDeg = get_line2line_angle(ps0.x, ps0.y, ps1.x, ps1.y, ps2.x, ps2.y, ps3.x, ps3.y) * 180.0/ M_PI;
					float deltaDeg = 90.0 - fabs(thetaDeg);
					if(fabs(deltaDeg)<20.0) { //Allow max 20 degrees misalignment from ideal 90 degrees intersection
            //TODO: some shit happening here with check of 90 angle on frame 737 in 2019_10_15_21h37m34s logs
            //std::cout << "thetaDeg=" << thetaDeg <<", fabs(deltaDeg) = " << fabs(deltaDeg) << std::endl;
            //Good corner detected
						//circle(img(), Point(i_x, i_y), 5, Scalar(0,0,255), -1);

            //cv::Point2f Ufa(((float) la[0]) / col_nb, ((float) la[1]) / row_nb);
            //cv::Point2f Vfa(((float) la[2]) / col_nb, ((float) la[3]) / row_nb);
            //cv::Point2f Ufb(((float) lb[0]) / col_nb, ((float) lb[1]) / row_nb);
            //cv::Point2f Vfb(((float) lb[2]) / col_nb, ((float) lb[3]) / row_nb);            
            //cv::Point2f C(((float) i_x) / col_nb, ((float) i_y) / row_nb);

            cv::Point2f Ufa(((float) la[0]), ((float) la[1])); //2019
            cv::Point2f Vfa(((float) la[2]), ((float) la[3])); //2019
            cv::Point2f Ufb(((float) lb[0]), ((float) lb[1])); //2019
            cv::Point2f Vfb(((float) lb[2]), ((float) lb[3])); //2019     
            cv::Point2f C(((float) i_x), ((float) i_y));       //2019     

            CornerCandiate cornerCandidate;
            cornerCandidate.ufa = Ufa;
            cornerCandidate.vfa = Vfa;
            cornerCandidate.ufb = Ufb;
            cornerCandidate.vfb = Vfb;            
            cornerCandidate.corner = C;
            cornerCandidate.score = my_norm(Ufa, Vfa) + my_norm(Ufb, Vfb);
            cornerCandidate.valid = true;
            cornerCandidates.push_back(cornerCandidate);

					}
				}
			}

		}
  }

  Benchmark::close("Checking lines for intersections");

  Benchmark::open("Merging corners");
  //Merge close corners by keeping only one corner in a cluster with largest line lengths. Not optimal but works
  bool changes_done = true;
  while(changes_done) {
    changes_done = false;
    for( size_t i = 0; i < cornerCandidates.size(); i++ ) {
      for( size_t j = i+1; j < cornerCandidates.size(); j++ ) {
        cv::Point2f c1 =  cornerCandidates[i].corner; 
        cv::Point2f c2 =  cornerCandidates[j].corner;
        if((cornerCandidates[i].valid)&&(cornerCandidates[j].valid)) {
          if(my_norm(c1,c2)<0.1) {
            //Corners are too close, filter one of them
            if(cornerCandidates[i].score>cornerCandidates[j].score) cornerCandidates[j].valid = false;
            else cornerCandidates[i].valid = false;
            changes_done = true;
          }
        }
      }
    }
  }  

  Benchmark::close("Merging corners");

  Benchmark::open("Filling data for robocup.cpp");
  //Data exchange vector for robocup.cpp
  loc_data_vector.clear();

  #if USE_LINES
  //First stage - fill data exchange vector with filtered line candidates (not corners)
  for( size_t i = 0; i < linesP.size(); i++ ) {
    Vec4i l = linesP[i];
    cv::Point2f p0, p1;
    try {
      p0 = cs->robotPosFromImg(l[0], l[1]);
      p1 = cs->robotPosFromImg(l[2], l[3]);
    }
      catch (const std::runtime_error& exc)
    {
      continue;
    }
    
    //Add only lines with appropriate length.
    //This will filter central circle segments and short sides of goal zone
    //std::cout << "Found line length = " << my_norm(p0,p1) << "m" << std::endl;
    float min_pixel_length = 175;
    float min_real_length = Constants::field.goal_area_length*1.1;
    
    if( my_norm(p0,p1) < min_real_length) continue;
    if( my_norm(cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3])) < min_pixel_length) continue;
    //cv::Point2f Uf(((float) l[0]) / col_nb, ((float) l[1]) / row_nb);
    cv::Point2f Uf(((float) l[0]), ((float) l[1])); //2019
    //cv::Point2f Vf(((float) l[2]) / col_nb, ((float) l[3]) / row_nb);      
    cv::Point2f Vf(((float) l[2]), ((float) l[3])); //2019

    //Creatin WhiteLinesData object, it can contains one or two lines and zero or one corner. 
    //For now add only lines
    float dummy_quality = 1.0;
    WhiteLinesData whiteLinesData = WhiteLinesData();    
    whiteLinesData.max_dist_corner = 10.0;
    whiteLinesData.tolerance_angle_line = 20.0; //10.0;
    whiteLinesData.tolerance_angle_corner = 25.0; //15.0; 
    whiteLinesData.minimal_segment_length = 0.001; //0.3;
    if(l[1]>l[3]) whiteLinesData.pushPixLine(dummy_quality, std::pair<cv::Point2f,cv::Point2f>(Uf,Vf));
    else whiteLinesData.pushPixLine(dummy_quality, std::pair<cv::Point2f,cv::Point2f>(Vf,Uf));
    loc_data_vector.push_back(whiteLinesData);

    //Draw the line being pushed to localisation in bold yellow
    line( im, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,200,200), 3, LINE_AA);
  }
  #endif

  #if USE_CORNERS
  //Second stage - fill data exchange vector with filtered corner candidates
  for( size_t i = 0; i < cornerCandidates.size(); i++ ) {
    if(cornerCandidates[i].valid) {
      //For now localisation module accepts only L-shaped corners.
      //Angle from "a" line to "b" line should be clockwise
      //So let's convert found corner (it can be X-shape, L-shape or T-shape) to L-shape with proper line order
      cv::Point2f Ufa = cornerCandidates[i].ufa;
      cv::Point2f Vfa = cornerCandidates[i].vfa;
      cv::Point2f Ufb = cornerCandidates[i].ufb;
      cv::Point2f Vfb = cornerCandidates[i].vfb;      
      cv::Point2f C = cornerCandidates[i].corner;  

      //Cutting lengths of shortest segments in T/X-spaed corner to convert it to L-shaped 
      //Corner will be processed as Ua-C-Vb notation in WhiteLiesCornerobservation.cpp, so Ub/Va values are useless
      float norm_a1 = my_norm(Ufa,C);
      float norm_a2 = my_norm(Vfa,C);
      if(norm_a1>norm_a2) Vfa = C; else { Ufa = Vfa; Vfa = C; };

      float norm_b1 = my_norm(Ufb,C);
      float norm_b2 = my_norm(Vfb,C);
      if(norm_b2>norm_b1) Ufb = C; else { Vfb = Ufb; Ufb = C; };
      
      //Angle from Ufa-C vector to Vfb-C vector should be anti-clocwise, 
      //overwise calculations of corner orientation in WhiteLiesCornerobservation.cpp will be wrong.
      //So let's correct each corner by cross product (=vector multiplication) z-component sign check
      cv::Point2f a = Ufa-C;
      cv::Point2f b = Vfb-C;
      double cross_product_z_component = a.x * b.y - b.x * a.y;
      if(cross_product_z_component>0) {
        //We need to swap Ufa and Vfb
        cv::Point2f t;
        t = Vfb;
        Vfb = Ufa;
        Ufa = t;
      }
      
      //Drawing raw corner (without conversion to L-shape)
      line( im, Point(cornerCandidates[i].ufa.x, cornerCandidates[i].ufa.y), 
                Point(cornerCandidates[i].vfa.x, cornerCandidates[i].vfa.y), Scalar(0,255,0), 3);
      line( im, Point(cornerCandidates[i].ufb.x, cornerCandidates[i].ufb.y), 
                Point(cornerCandidates[i].vfb.x, cornerCandidates[i].vfb.y), Scalar(0,255,0), 3);                
      circle(im, Point(cornerCandidates[i].corner.x, cornerCandidates[i].corner.y), 5, Scalar(0,0,255), -1); 

      //Drawing corner with conversion to L-shape
      //Angle from red to pink should be clockwise
      line( im, Point(Ufa.x, Ufa.y), 
                Point(C.x, C.y), Scalar(0,0,255), 2);
      line( im, Point(Vfb.x, Vfb.y), 
                Point(C.x, C.y), Scalar(0,128,255), 2);                
      
            
      //Creatin WhiteLinesData object, it can contains one or two lines and zero or one corner. Only corners is used by now
      

      float dummy_quality = 1.0;
      WhiteLinesData whiteLinesData = WhiteLinesData();        
      whiteLinesData.max_dist_corner = 10.0;
      whiteLinesData.tolerance_angle_line = 20.0; //10.0;
      whiteLinesData.tolerance_angle_corner = 25.0; //15.0; 
      whiteLinesData.minimal_segment_length = 0.001; //0.3;
      whiteLinesData.pushPixLine(dummy_quality, std::pair<cv::Point2f,cv::Point2f>(Ufa,Vfa));
      whiteLinesData.pushPixLine(dummy_quality, std::pair<cv::Point2f,cv::Point2f>(Ufb,Vfb));
      whiteLinesData.addPixCorner(C); //This will automatically set has_corner to true
      loc_data_vector.push_back(whiteLinesData);
    }
  }
  #endif

  Benchmark::close("Filling data for robocup.cpp");

  Benchmark::open("computeTransformations");  

  //Strategic TODO: one more pass of HoughP with shorter lines and strict length and angle check for penaty mark detection?

  //Strategic TODO: dewarp points and do RANSAC circle for field center detection

  // Drawing horizon for debug
  /*
  //TODO: bring it back in 2019 release
  cv::Point horizon1, horizon2;
  horizon1.x = 0;
  horizon2.x = source.cols - 1;
  horizon1.y = cs->getPixelYtAtHorizon(horizon1.x, source.cols, source.rows);
  horizon2.y = cs->getPixelYtAtHorizon(horizon2.x, source.cols, source.rows);
  cv::line(im, horizon1, horizon2, cv::Scalar(128, 128, 0), 2);
  */

  //std::cout << "-------------- WHITE LINES computeTransformations" << std::endl;
  for(int i=0;i<loc_data_vector.size();i++) {
    loc_data_vector[i].computeTransformations(&getCS(), true);
  }

  img() = im;
  //im.copyTo(img());
  //std::cout << "-------------- WHITE LINES PROCRESS() EXIT ------------" << std::endl;
  Benchmark::close("computeTransformations");
    
}

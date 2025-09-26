#include "SourceROS.hpp"
#include "Filters/Pipeline.hpp"
#include <rhoban_utils/util.h>
#include "RhIO.hpp"
#include "rhoban_utils/timing/benchmark.h"
#include <rhoban_utils/logging/logger.h>

#include <json/json.h>

namespace Vision {
namespace Filters {
SourceROS::SourceROS() : Source("SourceROS") {
  last_retrieval_success = TimeStamp::fromMS(((double)ros::Time::now().toNSec() / 1000000.0));
}
SourceROS::SourceROS(const std::string& source_name) : Source(source_name) {
  last_retrieval_success = TimeStamp::fromMS(((double)ros::Time::now().toNSec() / 1000000.0));
  generateDistortMap();
}

SourceROS::~SourceROS() {}

Source::Type SourceROS::getType() const { return Type::Online; }

std::string SourceROS::getClassName() const { return "SourceROS"; }

int SourceROS::expectedDependencies() const { return 0; }

void SourceROS::setRobotName(const std::string& robot_name) { robotName = robot_name; }

// Left camera is main
void SourceROS::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
  Benchmark::open("Processing captured image");
  // Show elapsed time since last call
  TimeStamp now = TimeStamp::fromMS(((double)ros::Time::now().toNSec() / 1000000.0));
  // TODO require to store lastRetrievalAttempt
  //     - (accumulation on elapsed when failing retrieval)
  double elapsed = diffMs(last_retrieval_success, now);
  elapsed_from_synch_ms = elapsed_from_synch_ms + elapsed;
  // std::cout << "elapsed_from_synch_ms=" << elapsed_from_synch_ms << std::endl;

  // TODO set as a json parameter?
  if ((elapsed_from_synch_ms > 10000) | first_run) {
    if (first_run) {
      first_run = false;
      last_ts = TimeStamp::fromMS(0);
    }

    elapsed_from_synch_ms = 0;
  }
  // Grab frame from camera topic
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    //cout << "Camera got image" << std::endl;
    
    //Doing distortion to simulate wideangle camera, since webots built-in spherical model and lens distortion model both not correct
    //cv::Mat tmp = cv_ptr->image;
    //remap(cv_ptr->image, cv_ptr->image, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));    
    
    img() = cv_ptr->image;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    return;
  }

  // Retrieve timestamp
  ros::Time ts = msg->header.stamp;

  double image_ts_ms = (double)ts.toNSec() / 1000000.0;  // Spinnaket timestamp is in nanoseconds

  // cout << "Image timestamp from ROS: " << image_ts_ms / 1000.0 << endl;

  // double normalized_frame_ts = image_ts_ms + ts_delta;
  /*double normalized_frame_ts = image_ts_ms;

  double now_ms = (double)ros::Time::now().toNSec() / 1000000.0;

  // printf("SourceROS: %5f\n", normalized_frame_ts);

  // Never allow to publish images more recent than current time!
  double frame_age_ms = now_ms - normalized_frame_ts;
  if (frame_age_ms < 0) {
    ROS_ERROR("%s::process: frame is dated from %5f ms in the future -> refused", deviceName, -frame_age_ms);
  }

  // getPipeline()->setTimestamp(rhoban_utils::TimeStamp::now()); //Dumb variant without timestamps processing
  */

  frame_ts = TimeStamp::fromMS(image_ts_ms);
  // cout << "Image timestamp CONVERTED from ROS: " << frame_ts.getTimeSec() << endl;
  // Processing order:
  // SourceRaw (Left, sync slave), timestamp for pipeline
  // SourceRaw2 (Right, sync master)

  if (isStereo) {
    if (!isRight) {
      // Left cam is a stereo sync slave and being inited first in pipline (!)
      // so lets put strictly it's timestamp to pipeline.
      // Checks shows no jumping on birdview with moving head when left cam timestamp is used
      // std::cout << "Updatinig pipeline timestamp stereo" << std::endl;
      getPipeline()->setTimestamp(frame_ts);
    } else {
      // std::cout << "Updatinig pipeline timestamp stereo right" << std::endl;
      // double stereoDeltaMs = diffMs(frame_ts, getPipeline()->getTimestamp());
      // std::cout << "stereoDeltaMs=" << stereoDeltaMs << std::endl;
    }
  } else {
    // std::cout << "Updatinig pipeline timestamp" << std::endl;
    getPipeline()->setTimestamp(frame_ts);
  }
  /*
  double elapsed_ms = diffMs(last_ts, frame_ts);
  if (elapsed_ms <= 0) {
    // updateRhIO();
    ROS_ERROR("%s:: Invalid elapsed time: %5f (Elapsed from sync %5f)", deviceName, elapsed_ms, elapsed_from_synch_ms);
  } else if (elapsed_ms > 500) {
    ROS_WARN("%s:: Elapsed time: %5f ms", deviceName, elapsed_ms);
  }
  last_ts = TimeStamp::fromMS(normalized_frame_ts);
  */
  Benchmark::close("Processing captured image");
}

void SourceROS::startCamera() {
  cout << deviceName << "::startCamera()" << std::endl;
  init_in_progress = true;
  is_capturing = true;
  init_in_progress = false;


}

void SourceROS::endCamera() {
  cout << deviceName << "::endCamera()" << std::endl;
  is_capturing = false;
  init_in_progress = false;
}
void SourceROS::process() {
  if (!is_capturing) {
    startCamera();
  }
}

void SourceROS::generateDistortMap() {

  //cv::Size size(1440, 1080);   //TODO: get rid of hardcoded size
  cv::Size size(720, 540);

  map1.create( size, CV_32FC1 ); //TODO: CV_16SC2 is faster, change according to https://github.com/egonSchiele/OpenCV/blob/master/modules/imgproc/src/undistort.cpp
  map2.create( size, CV_32FC1 );

  double p1 = 0;
  double p2 = 0;

  double cx = size.width / 2;
  double cy = size.height / 2;

  double k[8] = {k1, k2, p1, p2, k3, 0, 0, 0};
  double ifx = 1./fx;
  double ify = 1./fy;
  double x, y, x0, y0;
  int iters = 1;

  for( int i = 0; i < size.height; i++ ) {
    float* m1f = (float*)(map1.data + map1.step*i);
    float* m2f = (float*)(map2.data + map2.step*i);
    for( int j = 0; j < size.width; j++)
    {

      // Calculating inverse problem of distortion. Code taken from OpenCV/blob/master/modules/imgproc/src/undistort.cpp
      /*x = j;
      y = i;
      x0 = x = (x - cx)*ifx;
      y0 = y = (y - cy)*ify;      
      for( size_t t = 0; t < iters; t++ )
      {
          double r2 = x*x + y*y;
          double icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
          double deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x);
          double deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y;
          x = (x0 - deltaX)*icdist;
          y = (y0 - deltaY)*icdist;
      }
      double u = fx*x + cx;
      double v = fy*y + cy;      
      */

      // Calculating direct problem of distortion
      double x = (j - cx) / fx;
      double y = (i - cy) / fy;
      
      double x2 = x*x, y2 = y*y;
      double r2 = x2 + y2;
      double u = x * (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2);
      double v = y * (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2);
      u = fx*u + cx;
      v = fy*v + cy;      

      if(u < 0) u = -1;
      if(v < 0) v = -1;
      if(u > size.width) u = size.width;
      if(v > size.height) v = size.height;

      m1f[j] = (float)u;
      m2f[j] = (float)v;
    }
  } 

}

}  // namespace Filters
}  // namespace Vision
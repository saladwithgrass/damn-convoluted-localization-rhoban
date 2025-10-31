#include "Filters/Source/SourceSpinnaker.hpp"
#include "Filters/Pipeline.hpp"
#include <rhoban_utils/util.h>
#include "Utils/PtGreyExceptions.hpp"
#include "RhIO.hpp"
#include "rhoban_utils/timing/benchmark.h"
#include <rhoban_utils/logging/logger.h>

#include <json/json.h>

/*
Example MONO stup in vision_config.json:
        "filters" : [
            {
                "class name" : "SourceSpinnaker",
                "content" : {
                    "name" : "sourceRaw",
                    "warningExecutionTime": 0.05,
                    "shutter" : 5.0,
                    "gain" : 20.0,
                    "framerate" : 20.0,
		    "whitebalance_red" : 1.4,
		    "whitebalance_blue" : 3.0                    
                }
            },
*/

/*
Example STEREO stup in vision_config.json:            

        "filters" : [
            //Left cam is sync slave and should be inited before right
            {
                "class name" : "SourceSpinnaker",
                "content" : {
                    "name" : "sourceRaw",
                    "warningExecutionTime": 0.05,
                    "shutter" : 5.0,
                    "gain" : 20.0,
                    "framerate" : 20.0,
		                "whitebalance_red" : 1.4,
		                "whitebalance_blue" : 3.0,
		                "is_stereo" : true,
                    "camera_serial" : "19176743",
                    "is_right" : false
                }
            },

            //Right cam is sync master and will trigger left cam after starts to capture
            {
                "class name" : "SourceSpinnaker",
                "content" : {
                    "name" : "sourceRaw2",
                    "warningExecutionTime": 0.05,
                    "shutter" : 5.0,
                    "gain" : 20.0,
                    "framerate" : 20.0,
            		    "whitebalance_red" : 1.4,
		                "whitebalance_blue" : 3.0,
		                "is_stereo" : true,
                    "camera_serial" : "19176738",
                    "is_right" : true
                }
            },

            {
                "class name" : "StereoImgProc",
                "content" : {
                    "name" : "stereoImgProc",
                    "dependencies" : ["sourceRaw2", "sourceRaw", "YBirdview"]
                }
            },                 
*/

namespace Vision
{
namespace Filters
{
SourceSpinnaker::SourceSpinnaker() : Source("SourceSpinnaker")
{
  last_retrieval_success = TimeStamp::now();  

  pMemBuffer = (void*)std::malloc(memBufferTotalSize);

  std::string path = "calibration_stereo.json"; 
  
  //Reading stereo camera calibration params only for publishing it as camera_info for ROS built-in stereo_image_proc
  try{
    Json::Value json_content = rhoban_utils::file2Json(path);

    rhoban_utils::tryReadVector(json_content, "Kl", &Mvec[0]); 
    rhoban_utils::tryReadVector(json_content, "Kr", &Mvec[1]);

    rhoban_utils::tryReadVector(json_content, "Dl", &Dvec[0]);
    rhoban_utils::tryReadVector(json_content, "Dr", &Dvec[1]);

    rhoban_utils::tryReadVector(json_content, "Rl", &Rvec[0]);
    rhoban_utils::tryReadVector(json_content, "Rr", &Rvec[1]);

    rhoban_utils::tryReadVector(json_content, "Pl", &Pvec[0]);  
    rhoban_utils::tryReadVector(json_content, "Pr", &Pvec[1]);
    stereo_calibration_json_present = true;
  } catch (const rhoban_utils::JsonParsingError& error)  {
    cout << cameraDebugName << " :  calibration_stereo_.json not found or syntax error, skipping publishing camera_info to ROS" << std::endl;
    stereo_calibration_json_present = false;
  }

}

SourceSpinnaker::~SourceSpinnaker()
{
  std::free(pMemBuffer);

  //This code shoud be in endCamera() but is doesn't work in case of hot unplug
  if(pCam) {
    if( pCam->IsInitialized() ) pCam->DeInit();
    pCam = nullptr;
  }
  // Clear camera list before releasing system
  //camList.Clear();       

  // Release system
  if (system) system->ReleaseInstance();
}

Source::Type SourceSpinnaker::getType() const
{
  return Type::Online;
}

std::string SourceSpinnaker::getClassName() const
{
  return "SourceSpinnaker";
}

int SourceSpinnaker::expectedDependencies() const
{
  return 0;
}

void SourceSpinnaker::startCamera()
{
  cout << cameraDebugName << "::startCamera()" << std::endl;
  
  if(init_in_progress) return;
  


  // Retrieve singleton reference to system object
  if(system==nullptr) system = System::GetInstance(); 

  // Print out current library version
  /*
  const LibraryVersion spinnakerLibraryVersion = system->GetLibraryVersion();
  cout << "Spinnaker library version: " << spinnakerLibraryVersion.major << "." << spinnakerLibraryVersion.minor
        << "." << spinnakerLibraryVersion.type << "." << spinnakerLibraryVersion.build << endl
        << endl;
  */
  // Retrieve list of cameras from the system
  CameraList camList = system->GetCameras();

  const unsigned int numCameras = camList.GetSize();

  std::cout << "Number of cameras detected: " << numCameras << endl;

  if (numCameras == 0) {
    std::cout << cameraDebugName << " : No Spinaker cameras detected" << std::endl;    
    endCamera();
    //throw std::runtime_error("No Spinaker cameras detected");
  }
  
  // Strat vision porcessing only when camera are present for single cam setup, or both cameras are present for stereo setup
  int neededCameras = 1;
  if(is_stereo) neededCameras = 2;
  if (numCameras == neededCameras) { 

    init_in_progress = true;
    is_capturing = false; //Just to be sure    

    //Dirty hack for hotplug: if camera was power cycled during capture, we need to use new pCap object
    //otherwise "camera no longer valid and was removed from list" error will occur
    //Possible memory leaks here?
    if(pCam) pCam = nullptr;

    if(!is_stereo) {
      // Select first camera
      pCam = camList.GetByIndex(0);  
    } else {
      //Select camera by serial from json
      pCam = camList.GetBySerial(camera_serial);
    }
    
    if(!pCam) {
      std::cout << cameraDebugName << " : Camera ID is not that expected" << std::endl;   
      endCamera();
    } else {  
      try {
        pCam->SetBufferOwnership(BUFFER_OWNERSHIP_USER);
        
        pCam->SetUserBuffers(pMemBuffer, memBufferTotalSize);

        // Retrieve TL device nodemap and print device information
        INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

        //PrintDeviceInfo(nodeMapTLDevice); //For debug verbose print only

        // Initialize camera
        pCam->Init();

        // Retrieve GenICam nodemap
        INodeMap& nodeMap = pCam->GetNodeMap();

        // Configure trigger as in /usr/src/spinnaker/src/Trigger/Trigger.cpp 
        if(is_stereo) {
          ConfigureTrigger(nodeMap, is_right);
        } else {
          ConfigureTrigger(nodeMap, true);
        }

        // Activate chunk mode 
        CBooleanPtr ptrChunkModeActive = nodeMap.GetNode("ChunkModeActive");
        if (!IsAvailable(ptrChunkModeActive) || !IsWritable(ptrChunkModeActive))
        {
            cout << cameraDebugName << " : Unable to activate chunk mode. Aborting..." << endl << endl;
            //return -1;
        }
        ptrChunkModeActive->SetValue(true); 

        
        //
        // Enable all types of chunk data
        //
        
        NodeList_t entries;
        
        // Retrieve the selector node
        CEnumerationPtr ptrChunkSelector = nodeMap.GetNode("ChunkSelector");
        if (!IsAvailable(ptrChunkSelector) || !IsReadable(ptrChunkSelector))
        {
            cout << cameraDebugName << "Unable to retrieve chunk selector. Aborting..." << endl << endl;
            //return -1;
        }     

        // Retrieve entries
        ptrChunkSelector->GetEntries(entries);

        //cout << "Enabling entries..." << endl;

        for (size_t i = 0; i < entries.size(); i++)
        {
            // Select entry to be enabled
            CEnumEntryPtr ptrChunkSelectorEntry = entries.at(i);

            // Go to next node if problem occurs
            if (!IsAvailable(ptrChunkSelectorEntry) || !IsReadable(ptrChunkSelectorEntry))
            {
                continue;
            }

            ptrChunkSelector->SetIntValue(ptrChunkSelectorEntry->GetValue());

            //cout << "\t" << ptrChunkSelectorEntry->GetSymbolic() << ": ";

            // Retrieve corresponding boolean
            CBooleanPtr ptrChunkEnable = nodeMap.GetNode("ChunkEnable");

            // Enable the boolean, thus enabling the corresponding chunk data
            if (!IsAvailable(ptrChunkEnable))
            {
                //cout << "not available" << endl;
                //result = -1;
            }
            else if (ptrChunkEnable->GetValue())
            {
                //cout << "enabled" << endl;
            }
            else if (IsWritable(ptrChunkEnable))
            {
                ptrChunkEnable->SetValue(true);
                //cout << "enabled" << endl;
            }
            else
            {
                //cout << "not writable" << endl;
                //result = -1;
            }
        }            

        bindProperties();

        SetAcquisitionModeContinuous(nodeMap);

        pCam->BeginAcquisition();
        is_capturing = true;
        init_in_progress = false;

      }
      catch (Spinnaker::Exception& e)
      {
        init_in_progress = false;
        cout << "Error: " << e.what() << endl;
      }  
    }
  } else {
    if(numCameras > neededCameras) std::cout << cameraDebugName << " ERROR: too many cameras detected, not sure which to use" << std::endl;  
  }

  camList.Clear();

}

void SourceSpinnaker::endCamera()
{
  cout << cameraDebugName << "::endCamera()" << std::endl;
  

  //Dirty hack for hotplug: if camera was power cycled during capture, thre is no way to properly deinit it (will casue "camera is still streaming / camera is not started" errors)
  //So let's just forget about old camera instance and wait for new pCam object in startCamera()

  /*
  if(pCam) {

    // End acquisition
    if(is_capturing) {
    //if ( pCam->IsStreaming() ) {
      cout << "Spinnaker: calling EndAcquisition" << std::endl;
      pCam->EndAcquisition(); //Will throw Spinnaker: Camera is not started. [-1002] when called after hot unplug
      cout << "Spinnaker: EndAcquisition OK" << std::endl;
    }

    // Deinitialize camera
    if( pCam->IsInitialized() ) {
      cout << "Spinnaker: calling DeInit" << std::endl;
      pCam->DeInit();
      cout << "Spinnaker: DeInit OK" << std::endl;
    }

    //if( !(pCam->IsStreaming()) ) pCam->DeInit();
    pCam = nullptr; 
    cout << "Spinnaker: pCam = nullptr OK" << std::endl;
  }

  // Clear camera list before releasing system
  //camList.Clear();       

  // Release system
  if (system != nullptr) system->ReleaseInstance();
  */ 
  is_capturing = false;
  init_in_progress = false;
}

// The camera clock crystal, a CITIZEN CS10‐25.000MABJTR has a drift specification of ± 50 PPM
// Which means that after 20 minutes, we can expect the clock to drift +-60ms.
//=> If we want a ~1ms precision, we should call this function at least every 24 seconds.
double SourceSpinnaker::measureTimestampDelta()
{
  // This function costs ~3ms on the fitlet.

  /**How to estimate the delay between the capture moment and the image
   * reception moment? Sure, we have a precise, image embedded timsestamp, but
   * the clock of the camera and the clock of the PC are not synchronized. A
   * decent enough solution is to request the timestamp directly to the camera
   * (and not the value embedded in an image). Approximation : if t0 is the
   * local timestamp when we sent the order, and t1 is the local timestap when
   * we received the camera timestamp, then the received timestamp matches
   * (t0+t1)/2.
   */

  TimeStamp t0 = TimeStamp::now();
  
  // We need to ask the camera to latch (=flush) the timestamp
  CCommandPtr ptrTimestampLatch = pCam->GetNodeMap().GetNode("TimestampLatch");
  if (!IsAvailable(ptrTimestampLatch))
  {
      cout << cameraDebugName << " : Unable to call Time Stamp Latch (node retrieval). Aborting..." << endl;
      //return -1;
  }
  ptrTimestampLatch->Execute();

  TimeStamp t1 = TimeStamp::now();
  double elapsed = diffMs(t0, t1);

  // We can read the latched value now
  CIntegerPtr ptrTimestampLatchValue = pCam->GetNodeMap().GetNode("TimestampLatchValue");
  if (!IsAvailable(ptrTimestampLatchValue) || !IsReadable(ptrTimestampLatchValue))
  {
      cout << cameraDebugName << " : Unable to read Time Stamp Latch Value (node retrieval). Aborting.." << endl;
      //return -1;
  }
  int64_t timestampLatchValue = ptrTimestampLatchValue->GetValue();

  // From latched timestamp to milliseconds. Spinaker timestamp is in nanoseconds
  //double low = bufferLow / (double)125000;
  //double high = bufferHigh * (((unsigned long long int)1 << 32) / (double)125000);
  //double latchedMs = high + low;
  double latchedMs = (double)timestampLatchValue / 1000000.0;
  //latchedMs = fmod(latchedMs, 128000.0);
  double localTimestamp = t0.getTimeMS() + elapsed / 2;
  double delta = localTimestamp - latchedMs;

  return delta;
}

void SourceSpinnaker::process()
{

  if(!ros_inited) {
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    if(is_right) {
      pub_img = it.advertise("camera/right/image_raw", 1); 
      pub_camerainfo = nh.advertise<sensor_msgs::CameraInfo>("camera/right/camera_info", 1);        
    } else {
      pub_img = it.advertise("camera/left/image_raw", 1);
      pub_camerainfo = nh.advertise<sensor_msgs::CameraInfo>("camera/left/camera_info", 1);        
    }
    ros_inited = true;
  }

  TimeStamp processing_start = TimeStamp::now();

  if(!is_capturing) 
  {
    startCamera();
  }
  else 
  {


    // Import parameters from rhio and apply them if necessary
    Benchmark::open("Import from RhIO");
    importPropertiesFromRhIO();
    Benchmark::close("Import from RhIO");

    // Apply wished capture properties on spinnaker only once per second, because this call is very time-comsuming
    applyPropertiesDivider++;
    if(applyPropertiesDivider >= framerate) {
      applyPropertiesDivider = 0;
      Benchmark::open("Apply capture settings");
      applyWishedProperties();
      Benchmark::close("Apply capture settings");      
    }

    // Show elapsed time since last call
    TimeStamp now = TimeStamp::now();
    // TODO require to store lastRetrievalAttempt
    //     - (accumulation on elapsed when failing retrieval)
    double elapsed = diffMs(last_retrieval_success, now);
    elapsed_from_synch_ms = elapsed_from_synch_ms + elapsed;
    //std::cout << "elapsed_from_synch_ms=" << elapsed_from_synch_ms << std::endl; 

    // TODO set as a json parameter?
    if ((elapsed_from_synch_ms > 10000) | first_run)
    {
      if (first_run)
      {
        first_run = false;
        last_ts = TimeStamp::fromMS(0);
      }

      elapsed_from_synch_ms = 0;
      // Re-synch the pc timestamp with the camera timestamp
      ts_delta = measureTimestampDelta();
    }

    try {
      // Grab frame from camera
      Benchmark::open("Waiting for a new frame");
      ImagePtr pImage = pCam->GetNextImage(500*1000*1000); //Spinnaker timeout is in nanoseconds 
      //ImagePtr pImage = pCam->GetNextImage(1000); //Spinnaker timeout is in nanoseconds 
      Benchmark::close("Waiting for a new frame");

      if (pImage->IsIncomplete())
      {
          // Retrieve and print the image status description
          cout << "Image incomplete: " << Image::GetImageStatusDescription(pImage->GetImageStatus())
                << "..." << endl
                << endl;
      }
      else
      {  
        Benchmark::open("Processing captured image");
        last_retrieval_success = now;
        
        //Raw captured image is in BayerRG8 format
        ImagePtr pResultImage = pImage->Convert(PixelFormat_BGR8);

        const size_t width = pResultImage->GetWidth();
        const size_t height = pResultImage->GetHeight();
        unsigned int XPadding = pResultImage->GetXPadding();
        unsigned int YPadding = pResultImage->GetYPadding();      
        cv::Mat tmp_img = cv::Mat(height + YPadding, width + XPadding, CV_8UC3, pResultImage->GetData(), pResultImage->GetStride());
        
        if(dirtyHackFakeSingleImage) {
          img() = cv::imread("/home/rhoban/fakeImage.png",  cv::IMREAD_COLOR);
        } else {
          img() = tmp_img.clone(); //Cloning is necessary here       
        }


        // Retrieve timestamp
        ChunkData chunkData = pImage->GetChunkData();
        uint64_t ts = chunkData.GetTimestamp();

        double image_ts_ms = (double)ts/1000000.0; //Spinnaket timestamp is in nanoseconds

        //cout << "\tTimestamp: " << ts << endl;

        double normalized_frame_ts = image_ts_ms + ts_delta;

        double now_ms = TimeStamp::now().getTimeMS();

        // Never allow to publish images more recent than current time!
        double frame_age_ms = now_ms - normalized_frame_ts;
        if (frame_age_ms < 0)
        {
          std::ostringstream oss;
          oss << cameraDebugName << "::process: frame is dated from " << (-frame_age_ms) << " ms in the future -> refused";
          measureTimestampDelta();
          throw Utils::PtGreyException(oss.str());
        }
        if (frame_age_ms > 128000) //TODO: 128000 is the magic number from PtGrey, fix it for Spinnaker
        {
          std::ostringstream oss;
          oss << cameraDebugName << "::process: frame is dated from " << frame_age_ms << " ms in the past -> too old";
          measureTimestampDelta();
          throw Utils::PtGreyException(oss.str());
        }

        //getPipeline()->setTimestamp(rhoban_utils::TimeStamp::now()); //Dumb variant without timestamps processing

        frame_ts = TimeStamp::fromMS(normalized_frame_ts);
        //Processing order:
        //SourceRaw (Left, sync slave), timestamp for pipeline
        //SourceRaw2 (Right, sync master) 

        if(is_stereo) {
          if(!is_right) {
            //Left cam is a stereo sync slave and being inited first in pipline (!)
            //so lets put strictly it's timestamp to pipeline. 
            //Checks shows no jumping on birdview with moving head when left cam timestamp is used
            getPipeline()->setTimestamp(frame_ts);  
          } else {
            //double stereoDeltaMs = diffMs(frame_ts, getPipeline()->getTimestamp());
            //std::cout << "stereoDeltaMs=" << stereoDeltaMs << std::endl;
          }
        } else {
          getPipeline()->setTimestamp(frame_ts);
        }


        double elapsed_ms = diffMs(last_ts, frame_ts);
        if (elapsed_ms <= 0)
        {
          //updateRhIO();
          std::ostringstream oss;
          oss << cameraDebugName << "Invalid elapsed time: " << elapsed_ms << " (Elapsed from sync " << elapsed_from_synch_ms << ")";
          throw Utils::PtGreyException(oss.str());
        }
        else if (elapsed_ms > 500)
        {
          std::cout << cameraDebugName << ":: Warning: Elapsed time: " << elapsed_ms << " ms" << std::endl;
        }
        last_ts = TimeStamp::fromMS(normalized_frame_ts);          
        Benchmark::close("Processing captured image");

        if(publishToROS) {
          Benchmark::open("Publish to ROS");

          //Publishing image
          //ros::Time ros_timestamp = ros::Time::now();
          ros::Time ros_timestamp;
          ros_timestamp.fromNSec(normalized_frame_ts*1000000.0);
          cv::pyrDown(tmp_img, tmp_img);
          sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", tmp_img).toImageMsg();
          img_msg->header.stamp = ros_timestamp;
          //img_msg->time = ros_timestamp;    
          if(is_right) img_msg->header.frame_id = "cam_right_optical_frame";
          else img_msg->header.frame_id = "cam_left_optical_frame";

          pub_img.publish(img_msg);
          
          //Publishing camera_info
          //This stuff is needed only to be able to run ros stereo_image_proc node (http://wiki.ros.org/stereo_image_proc) for tests
          if(stereo_calibration_json_present) {
          
            int j = is_right ? 1 : 0;
            sensor_msgs::CameraInfoPtr ci_msg(new sensor_msgs::CameraInfo());
            ci_msg->header.stamp = ros_timestamp;
            //img_msg->time = ros_timestamp;         
            int image_width = tmp_img.cols;
            int image_height = tmp_img.rows;
            //std::string distortion_model = "plumb_bob";
            std::string distortion_model = "rational_polynomial";
            if(is_right) ci_msg->header.frame_id = "cam_right_optical_frame";
            else ci_msg->header.frame_id = "cam_left_optical_frame";
            ci_msg->height = image_height;
            ci_msg->width = image_width;
            ci_msg->distortion_model = distortion_model;
            
            if(distortion_model == "rational_polynomial") {
              //Extend distortion vec size to 8 elements required for rational_polynomial
              //Needed when calibration in json was done for plumb_bob and need to be rerun for rational_polynomial
              while(Dvec[j].size() < 8) Dvec[j].push_back(0.0);
            }    
            ci_msg->D = Dvec[j];
            // intrinsic coefficients
            for (int count = 0; count<Mvec[j].size();count++){
                ci_msg->K[count] = Mvec[j][count];
            }
            // Rectification matrix
            ci_msg->R = {
                Rvec[j][0], Rvec[j][1], 
                Rvec[j][2], Rvec[j][3], 
                Rvec[j][4], Rvec[j][5], 
                Rvec[j][6], Rvec[j][7], 
                Rvec[j][8]};
            // Projection/camera matrix
            ci_msg->P = {
                Pvec[j][0], Pvec[j][1], 
                Pvec[j][2], Pvec[j][3], 
                Pvec[j][4], Pvec[j][5], 
                Pvec[j][6], Pvec[j][7], 
                Pvec[j][8], Pvec[j][9], 
                Pvec[j][10], Pvec[j][11]};
            
            pub_camerainfo.publish(ci_msg);
          }
          Benchmark::close("Publish to ROS");
        }

      }

      // Release image
      // Images retrieved directly from the camera (i.e. non-converted
      // images) need to be released in order to keep from filling the
      // buffer.
      pImage->Release(); 

    }
    catch (Spinnaker::Exception& e)
    {
      Benchmark::close("Waiting for a new frame");
      cout << "Error: " << e.what() << endl;
      is_capturing = false;
      endCamera();
    }
  }

  TimeStamp processing_end = TimeStamp::now();
  double processing_ms = diffMs(processing_start, processing_end);
  //std::cout << "Spinnaker camera: frame grabbing time: " << processing_ms << " ms" << std::endl;
  
}

void SourceSpinnaker::fromJson(const Json::Value& v, const std::string& dir_name)
{
  Source::fromJson(v, dir_name);
  rhoban_utils::tryRead(v, "shutter", &shutter);
  rhoban_utils::tryRead(v, "gain", &gain);
  rhoban_utils::tryRead(v, "framerate", &framerate);
  rhoban_utils::tryRead(v, "whitebalance_blue", &whitebalance_blue);
  rhoban_utils::tryRead(v, "whitebalance_red", &whitebalance_red);
  rhoban_utils::tryRead(v, "camera_serial", &camera_serial);
  rhoban_utils::tryRead(v, "is_right", &is_right);
  rhoban_utils::tryRead(v, "is_stereo", &is_stereo);
  
  std::cout << "shutter=" << shutter << std::endl;
  std::cout << "gain=" << gain << std::endl;
  std::cout << "framerate=" << framerate << std::endl;
  std::cout << "whitebalance_blue=" << whitebalance_blue << std::endl;
  std::cout << "whitebalance_red=" << whitebalance_red << std::endl;
  std::cout << "camera_serial=" << camera_serial << std::endl;
  std::cout << "is_right=" << is_right << std::endl;
  std::cout << "is_stereo=" << is_stereo << std::endl;

  //N.B.: If using stereo head in mono setup - use left camera
  if(!is_stereo) cameraDebugName = "SourceSpinnakerSingle";
  else if(is_right) cameraDebugName = "SourceSpinnakerRight";
  else cameraDebugName = "SourceSpinnakerLeft";
}

double SourceSpinnaker::getFrameRate() const
{
  return framerate;
}

Json::Value SourceSpinnaker::toJson() const
{
  Json::Value v = Source::toJson();
  v["device_name"] = device_name;
  return v;
}

int SourceSpinnaker::SetAcquisitionModeContinuous(INodeMap& nodeMap)
{
  int result = 0;
  try
  {
    //
    // Set acquisition mode to continuous
    //
    // *** NOTES ***
    // Because the example acquires and saves 10 images, setting acquisition
    // mode to continuous lets the example finish. If set to single frame
    // or multiframe (at a lower number of images), the example would just
    // hang. This would happen because the example has been written to
    // acquire 10 images while the camera would have been programmed to
    // retrieve less than that.
    //
    // Setting the value of an enumeration node is slightly more complicated
    // than other node types. Two nodes must be retrieved: first, the
    // enumeration node is retrieved from the nodemap; and second, the entry
    // node is retrieved from the enumeration node. The integer value of the
    // entry node is then set as the new value of the enumeration node.
    //
    // Notice that both the enumeration and the entry nodes are checked for
    // availability and readability/writability. Enumeration nodes are
    // generally readable and writable whereas their entry nodes are only
    // ever readable.
    //
    // Retrieve enumeration node from nodemap
    CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
    if (!IsAvailable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode))
    {
        cout << cameraDebugName << " : Unable to set acquisition mode to continuous (enum retrieval). Aborting..." << endl << endl;
        return -1;
    }

    // Retrieve entry node from enumeration node
    //CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("SingleFrame");
    CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
    if (!IsAvailable(ptrAcquisitionModeContinuous) || !IsReadable(ptrAcquisitionModeContinuous))
    {
        cout << cameraDebugName << " : Unable to set acquisition mode to continuous (entry retrieval). Aborting..." << endl << endl;
        return -1;
    }

    // Retrieve integer value from entry node
    const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

    // Set integer value from entry node as new value of enumeration node
    ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

    cout << cameraDebugName << " : Acquisition mode set to continuous..." << endl; 
  }
  catch (Spinnaker::Exception& e)
  {
      cout << "Error: " << e.what() << endl;
      return -1;
  }
  return 0;
}

// This function prints the device information of the camera from the transport
// layer; please see NodeMapInfo example for more in-depth comments on printing
// device information from the nodemap.
int SourceSpinnaker::PrintDeviceInfo(INodeMap& nodeMap)
{
    cout << endl << "*** DEVICE INFORMATION ***" << endl << endl;

    try
    {
        FeatureList_t features;
        const CCategoryPtr category = nodeMap.GetNode("DeviceInformation");
        if (IsAvailable(category) && IsReadable(category))
        {
            category->GetFeatures(features);

            for (auto it = features.begin(); it != features.end(); ++it)
            {
                const CNodePtr pfeatureNode = *it;
                cout << pfeatureNode->GetName() << " : ";
                CValuePtr pValue = static_cast<CValuePtr>(pfeatureNode);
                cout << (IsReadable(pValue) ? pValue->ToString() : "Node not readable");
                cout << endl;
            }
        }
        else
        {
            cout << "Device control information not available." << endl;
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}       

void SourceSpinnaker::bindProperties()
{
  std::string filter_path = rhio_path + getName();

  //std::string property_path = filter_path + "/";
  RhIO::IONode& node = RhIO::Root.child(filter_path);
  node.newFloat("shutter")->defaultValue(shutter);    
  node.newFloat("gain")->defaultValue(gain); 
  node.newFloat("framerate")->defaultValue(framerate); 
  node.newFloat("whitebalance_blue")->defaultValue(whitebalance_blue); 
  node.newFloat("whitebalance_red")->defaultValue(whitebalance_red); 
  node.newBool("dirtyHackFakeSingleImage")->defaultValue(dirtyHackFakeSingleImage);
  node.newBool("publishToROS")->defaultValue(publishToROS);
  
}

void SourceSpinnaker::importPropertiesFromRhIO()
{
  std::string filter_path = rhio_path + getName();

  RhIO::IONode& node = RhIO::Root.child(filter_path);
  shutter = node.getValueFloat("shutter").value;
  gain = node.getValueFloat("gain").value;
  framerate = node.getValueFloat("framerate").value;
  whitebalance_blue = node.getValueFloat("whitebalance_blue").value;
  whitebalance_red = node.getValueFloat("whitebalance_red").value;
  dirtyHackFakeSingleImage = node.getValueBool("dirtyHackFakeSingleImage").value;
  publishToROS = node.getValueBool("publishToROS").value;

}

int SourceSpinnaker::applyWishedProperties(void)
{

  try
  {
    double exposureTimeToSet = shutter*1000.0; //in PtGrey it is in ms, in Spinnaker it's in us

    // Turn off auto exposure
    if(pCam->ExposureAuto.GetValue() != Spinnaker::ExposureAutoEnums::ExposureAuto_Off) {
      std::cout << "Setting ExposureAuto_Off" << std::endl;
      pCam->ExposureAuto.SetValue(Spinnaker::ExposureAutoEnums::ExposureAuto_Off);
    }
    //Set exposure mode to "Timed"
    if(pCam->ExposureMode.GetValue() != Spinnaker::ExposureModeEnums::ExposureMode_Timed) {
      std::cout << "Setting ExposureMode_Timed" << std::endl;
      pCam->ExposureMode.SetValue(Spinnaker::ExposureModeEnums::ExposureMode_Timed);
    }
    //Set absolute value of shutter exposure time    
    if(fabs(pCam->ExposureTime.GetValue() - exposureTimeToSet) > exposureTimeToSet*0.01) {
      std::cout << "Setting shutter" << std::endl;
      pCam->ExposureTime.SetValue(exposureTimeToSet); //in PtGrey it is in ms, in Spinnaker it's in us
    }

    //Turn auto gain off
    if(pCam->GainAuto.GetValue() != Spinnaker::GainAutoEnums::GainAuto_Off) {
      std::cout << "Setting GainAuto_Off" << std::endl;
      pCam->GainAuto.SetValue(Spinnaker::GainAutoEnums::GainAuto_Off);
    }
    if(fabs(pCam->Gain.GetValue() - gain) > gain*0.01) {
      std::cout << "Setting gain" << std::endl;
      pCam->Gain.SetValue(gain);
    }


    // Turn on frame rate control
    if(pCam->AcquisitionFrameRateEnable.GetValue() != true) {
      std::cout << "Setting AcquisitionFrameRateEnable" << std::endl;
      pCam->AcquisitionFrameRateEnable.SetValue(true);
    }
    if(fabs(pCam->AcquisitionFrameRate.GetValue() - framerate) > framerate*0.01) {
      std::cout << "Setting framerate" << std::endl;
      pCam->AcquisitionFrameRate.SetValue(framerate);
    }

    //Set auto white balance to off
    if(pCam->BalanceWhiteAuto.GetValue() != Spinnaker::BalanceWhiteAutoEnums::BalanceWhiteAuto_Off) {
      std::cout << "Setting BalanceWhiteAuto_Off" << std::endl;
      pCam->BalanceWhiteAuto.SetValue(Spinnaker::BalanceWhiteAutoEnums::BalanceWhiteAuto_Off);
    }

    CFloatPtr BalanceRatio = pCam->GetNodeMap().GetNode("BalanceRatio");

    //Select blue channel balance ratio
    pCam->BalanceRatioSelector.SetValue(Spinnaker::BalanceRatioSelectorEnums::BalanceRatioSelector_Blue);
    //Set the white balance blue channel
    if(fabs(BalanceRatio->GetValue() - whitebalance_blue) > whitebalance_blue*0.01) {
      std::cout << "Setting whitebalance_blue" << std::endl;
      BalanceRatio->SetValue(whitebalance_blue);
    }

    //Select blue channel balance ratio
    pCam->BalanceRatioSelector.SetValue(Spinnaker::BalanceRatioSelectorEnums::BalanceRatioSelector_Red);
    //Set the white balance red channel
    if(fabs(BalanceRatio->GetValue() - whitebalance_red) > whitebalance_red*0.01) {
      std::cout << "Setting whitebalance_red" << std::endl;
      BalanceRatio->SetValue(whitebalance_red);
    }

  }
  catch (Spinnaker::Exception& e)
  {
    cout << "Error: " << e.what() << endl;
    return -1;
  }
  
  return 0;
}

/*
Master setup:

cams[i].setEnumValue("LineSelector", "Line1"); //Both DONE
cams[i].setEnumValue("TriggerSource", "Software"); //both DONE
cams[i].setEnumValue("LineMode", "Output"); //Both DONE

cams[i].setEnumValue("TriggerMode", "On");  //Both DONE

Slave setup:
cams[i].setEnumValue("LineSelector", "Line2"); //Both DONE
cams[i].setEnumValue("TriggerSource", "Line2"); //Both DONE
cams[i].setEnumValue("LineMode", "Input"); //Both DONE

cams[i].setEnumValue("TriggerSelector", "FrameStart");
//cams[i].setFloatValue("TriggerDelay", 40.0);
cams[i].setEnumValue("TriggerOverlap", "ReadOut");
cams[i].setEnumValue("TriggerActivation", "RisingEdge"); 

cams[i].setEnumValue("TriggerMode", "On"); //Both DONE

*/

int SourceSpinnaker::ConfigureTrigger(INodeMap& nodeMap, bool is_master)
{
    int result = 0;

    cout << cameraDebugName << " *** CONFIGURING TRIGGER ***" << endl << endl;

    if (is_master)
    {
        cout << cameraDebugName << " : Software trigger chosen..." << endl;
    }
    else
    {
        cout << cameraDebugName << " : Hardware trigger chosen..." << endl;
    }

    try
    {
        //
        // Ensure trigger mode off
        //
        // *** NOTES ***
        // The trigger must be disabled in order to configure whether the source
        // is software or hardware.
        //
        CEnumerationPtr ptrTriggerMode = nodeMap.GetNode("TriggerMode");
        if (!IsAvailable(ptrTriggerMode) || !IsReadable(ptrTriggerMode))
        {
            cout << cameraDebugName << " : Unable to disable trigger mode (node retrieval). Aborting..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");
        if (!IsAvailable(ptrTriggerModeOff) || !IsReadable(ptrTriggerModeOff))
        {
            cout << cameraDebugName << " : Unable to disable trigger mode (enum entry retrieval). Aborting..." << endl;
            return -1;
        }

        ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());

        cout << cameraDebugName << " : Trigger mode disabled..." << endl;
        //-----------------------------------------------------------------------------------

        //Setting "LineSelector" to Line1(master) / Line2(slave)
        CEnumerationPtr ptrLineSelector = nodeMap.GetNode("LineSelector");
        if (!IsAvailable(ptrLineSelector) || !IsWritable(ptrLineSelector))
        {
            cout << cameraDebugName << " : Unable to set LineSelector (node retrieval). Aborting..." << endl;
            return -1;
        }
        if(is_master) {
          CEnumEntryPtr ptrLineSelectorLine1 = ptrLineSelector->GetEntryByName("Line1");
          if (!IsAvailable(ptrLineSelectorLine1) || !IsReadable(ptrLineSelectorLine1))
          {
              cout << cameraDebugName << " : Unable to set trigger mode (enum entry retrieval). Aborting..." << endl;
              return -1;
          }   
          ptrLineSelector->SetIntValue(ptrLineSelectorLine1->GetValue());       
        } else {
          CEnumEntryPtr ptrLineSelectorLine2 = ptrLineSelector->GetEntryByName("Line2");
          if (!IsAvailable(ptrLineSelectorLine2) || !IsReadable(ptrLineSelectorLine2))
          {
              cout << cameraDebugName << " : Unable to set trigger mode (enum entry retrieval). Aborting..." << endl;
              return -1;
          }   
          ptrLineSelector->SetIntValue(ptrLineSelectorLine2->GetValue());        
        }

        //Setting "LineMode" to Output(master) / Input(slave)
        CEnumerationPtr ptrLineMode = nodeMap.GetNode("LineMode");
        if (!IsAvailable(ptrLineMode) || !IsWritable(ptrLineMode))
        {
            cout << cameraDebugName << " : Unable to set LineMode (node retrieval). Aborting..." << endl;
            return -1;
        }
        if(is_master) {
          CEnumEntryPtr ptrLineModeOutput = ptrLineMode->GetEntryByName("Output");
          if (!IsAvailable(ptrLineModeOutput) || !IsReadable(ptrLineModeOutput))
          {
              cout << cameraDebugName << " : Unable to set ptrLineModeOutput (enum entry retrieval). Aborting..." << endl;
              return -1;
          }    
          ptrLineMode->SetIntValue(ptrLineModeOutput->GetValue());       
        } else {
          CEnumEntryPtr ptrLineModeInput = ptrLineMode->GetEntryByName("Input");
          if (!IsAvailable(ptrLineModeInput) || !IsReadable(ptrLineModeInput))
          {
              cout << cameraDebugName << " : Unable to set ptrLineModeInput (enum entry retrieval). Aborting..." << endl;
              return -1;
          }
          ptrLineMode->SetIntValue(ptrLineModeInput->GetValue());           
        }

        if(is_master==false) {
          //Slave additiom setup:
          
          //cams[i].setEnumValue("TriggerSelector", "FrameStart");
          CEnumerationPtr ptrTriggerSelector = nodeMap.GetNode("TriggerSelector");
          if (!IsAvailable(ptrTriggerSelector) || !IsWritable(ptrTriggerSelector))
          {
              cout << cameraDebugName << " : Unable to set TriggerSelector (node retrieval). Aborting..." << endl;
              return -1;
          }
          CEnumEntryPtr ptrFrameStart = ptrTriggerSelector->GetEntryByName("FrameStart");
          if (!IsAvailable(ptrFrameStart) || !IsReadable(ptrFrameStart))
          {
              cout << cameraDebugName << " : Unable to set ptrFrameStart (enum entry retrieval). Aborting..." << endl;
              return -1;
          }    
          ptrTriggerSelector->SetIntValue(ptrFrameStart->GetValue()); 

          ////cams[i].setFloatValue("TriggerDelay", 40.0);

          //cams[i].setEnumValue("TriggerOverlap", "ReadOut");
          CEnumerationPtr ptrTriggerOverlap = nodeMap.GetNode("TriggerOverlap");
          if (!IsAvailable(ptrTriggerOverlap) || !IsWritable(ptrTriggerOverlap))
          {
              cout << cameraDebugName << " : Unable to set ptrTriggerOverlap (node retrieval). Aborting..." << endl;
              return -1;
          }
          CEnumEntryPtr ptrReadOut = ptrTriggerOverlap->GetEntryByName("ReadOut");
          if (!IsAvailable(ptrReadOut) || !IsReadable(ptrReadOut))
          {
              cout << cameraDebugName << " : Unable to set ptrReadOut (enum entry retrieval). Aborting..." << endl;
              return -1;
          }    
          ptrTriggerOverlap->SetIntValue(ptrReadOut->GetValue());     

          //cams[i].setEnumValue("TriggerActivation", "RisingEdge"); 
          /*CEnumerationPtr ptrTriggerActivation = nodeMap.GetNode("TriggerActivation");
          if (!IsAvailable(ptrTriggerActivation) || !IsWritable(ptrTriggerActivation))
          {
              cout << "Unable to set ptrTriggerActivation (node retrieval). Aborting..." << endl;
              return -1;
          }
          CEnumEntryPtr ptrRisingEdge = ptrTriggerActivation->GetEntryByName("RisingEdge");
          if (!IsAvailable(ptrRisingEdge) || !IsReadable(ptrRisingEdge))
          {
              cout << "Unable to set ptrRisingEdge (enum entry retrieval). Aborting..." << endl;
              return -1;
          }    
          ptrTriggerActivation->SetIntValue(ptrRisingEdge->GetValue());              
          */

        }

        //
        // Select trigger source
        //
        // *** NOTES ***
        // The trigger source must be set to hardware or software while trigger
        // mode is off.
        //
        CEnumerationPtr ptrTriggerSource = nodeMap.GetNode("TriggerSource");
        if (!IsAvailable(ptrTriggerSource) || !IsWritable(ptrTriggerSource))
        {
            cout << cameraDebugName << " : Unable to set trigger mode (node retrieval). Aborting..." << endl;
            return -1;
        }

        if (is_master)
        {
            // Set trigger mode to software for master camera
            /*
            CEnumEntryPtr ptrTriggerSourceSoftware = ptrTriggerSource->GetEntryByName("Software");
            if (!IsAvailable(ptrTriggerSourceSoftware) || !IsReadable(ptrTriggerSourceSoftware))
            {
                cout << "Unable to set trigger mode (enum entry retrieval). Aborting..." << endl;
                return -1;
            }

            ptrTriggerSource->SetIntValue(ptrTriggerSourceSoftware->GetValue());

            cout << "Trigger source set to software..." << endl;
            */
        }
        else 
        {
            // Set trigger mode to hardware from Line2 for slave
            CEnumEntryPtr ptrTriggerSourceHardware = ptrTriggerSource->GetEntryByName("Line2");
            if (!IsAvailable(ptrTriggerSourceHardware) || !IsReadable(ptrTriggerSourceHardware))
            {
                cout << cameraDebugName << " : Unable to set trigger mode (enum entry retrieval). Aborting..." << endl;
                return -1;
            }

            ptrTriggerSource->SetIntValue(ptrTriggerSourceHardware->GetValue());

            cout << cameraDebugName << " : Trigger source set to hardware..." << endl;
        }

        //
        // Turn trigger mode on
        //
        // *** LATER ***
        // Once the appropriate trigger source has been set, turn trigger mode
        // on in order to retrieve images using the trigger.
        //
        if (is_master==false) {
          CEnumEntryPtr ptrTriggerModeOn = ptrTriggerMode->GetEntryByName("On");
          if (!IsAvailable(ptrTriggerModeOn) || !IsReadable(ptrTriggerModeOn))
          {
              cout << cameraDebugName << " : Unable to enable trigger mode (enum entry retrieval). Aborting..." << endl;
              return -1;
          }

          ptrTriggerMode->SetIntValue(ptrTriggerModeOn->GetValue());

        // TODO: Blackfly and Flea3 GEV cameras need 1 second delay after trigger mode is turned on

        cout << cameraDebugName << " : Trigger mode turned back on..." << endl << endl;
        }

    }
    catch (Spinnaker::Exception& e)
    {
        cout << cameraDebugName << " : Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}              

}  // namespace Filters
}  // namespace Vision
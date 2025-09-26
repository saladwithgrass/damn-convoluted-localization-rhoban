#pragma once

#include "Filters/Filter.hpp"
#include "Filters/Features/FeaturesProvider.hpp"

//#include <gflags/gflags.h>
#include <functional>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <algorithm>
#include <iterator>

#include <inference_engine.hpp>

#include <ngraph/ngraph.hpp>

#include <samples/ocv_common.hpp>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/dnn/dnn.hpp>
#include <cmath>

#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif

using namespace InferenceEngine;

namespace Vision
{
namespace Filters
{
struct DetectedObject : public cv::Rect {
    int class_id;
    std::string label;
    float confidence;
    

    bool operator <(const DetectedObject &s2) const {
        return this->confidence < s2.confidence;
    }
    bool operator >(const DetectedObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

struct Detection
    {
        int class_id;
        std::string label;
        float confidence;
        cv::Rect box;
        float x;
        float y;
    };

class YoloV5OpenVino : public Filter, public FeaturesProvider
{
public:

  YoloV5OpenVino();
  virtual std::string getClassName() const override;
  virtual int expectedDependencies() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

protected:
  virtual void process() override;
  virtual void setParameters() override;
  // void updateUsedClasses();


private:
  void init();
  cv::Mat formatYolov5(const cv::Mat &source);
  void detect(cv::Mat &image, std::vector<Detection> &outputs);
  // void readNetwork();
  // void prepareInputsOutputs();
  // void loadNetworkToDevice();
  // void createInferenceRequests();
  // void FrameToBlob(const cv::Mat &frame, InferRequest &inferRequest, const std::string &inputName);
  double sigmoid(double x);
  std::vector<int> getAnchors(int net_grid);
  // void parseOutput(const Blob::Ptr &blob,int net_grid,
  //   std::vector<cv::Rect>& o_rect,std::vector<float>& o_rect_cof, std::vector<int>& detected_ids);

  void NMSBoxes(const std::vector<cv::Rect>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold,
      std::vector<int>& indices,
      int limit);
  void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec);

  ParamInt debugLevel;
  std::map<std::string,ParamInt> isUsingFeature;
  ParamInt imSize;
  ParamFloat confBallThreshold;
  ParamFloat confPostThreshold;
  ParamFloat confRobotThreshold;
  ParamFloat nmsThreshold;
  ParamFloat scoreThreshold;
  std::string model_path;
  std::string labels_path;
  std::string device;
  bool useAutoResize = true;

  std::vector<std::string> labels;

  int numClasses = 0;

  std::string inputsName = "";
  std::string outputsName = "";

  size_t netInputHeight = 0;
  size_t netInputWidth = 0;


  Core ie;
  InferenceEngine::CNNNetwork net;
  InferenceEngine::ExecutableNetwork executableNet;
  //InferRequest::Ptr async_infer_request_next;
  InferRequest infer_request;

  OutputsDataMap outputs;
  InputsDataMap inputs;

  InferenceEngine::Blob::Ptr m_inputData;

  size_t m_numChannels = 0;
  size_t m_inputH = 0;
  size_t m_inputW = 0;
  size_t m_imageSize = 0;
  double min_conf_threshold;

  cv::Mat frame;
  cv::Mat next_frame;

  bool isLastFrame = false;
  bool isAsyncMode = false;  // execution is always started using SYNC mode
  bool isModeChanged = false;
  bool onStart = true;
  bool draw_result = false;

  static std::map<std::string, hl_monitoring::Field::POIType> stringToPOIEnum;  // TODO find better name ...
};

}  // namespace Filters
}  // namespace Vision

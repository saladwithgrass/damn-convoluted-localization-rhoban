#include "Filters/Custom/YoloV5OpenVino.hpp"
#include "CameraState/CameraState.hpp"
#include "Filters/Patches/PatchProvider.hpp"
#include "Utils/RotatedRectUtils.hpp"
#include "Utils/Interface.h"
#include "Utils/OpencvUtils.h"
#include "Utils/ROITools.hpp"
#include "rhoban_utils/timing/benchmark.h"

#include "rhoban_geometry/circle.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <utility>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <rhoban_utils/logging/logger.h>

#include <opencv2/dnn/dnn.hpp>

#include <hl_monitoring/field.h>

static rhoban_utils::Logger logger("YoloV5OpenVino");

using namespace std;
using namespace rhoban_geometry;
using ::rhoban_utils::Benchmark;


bool SortScorePairDescend(const std::pair<float, int>& pair1,
                          const std::pair<float, int>& pair2)
{
    return pair1.first > pair2.first;
}

double jaccardDistance(const cv::Rect& a, const cv::Rect& b) {
    double Aa = a.area();
    double Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<double>::epsilon()) {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    double Aab = (a & b).area();
    // distance = 1 - jaccard_index
    return 1.0 - Aab / (Aa + Ab - Aab);
}

float rectOverlap(const cv::Rect& a, const cv::Rect& b)
{
    return 1.f - static_cast<float>(jaccardDistance(a, b));
}

namespace Vision
{
namespace Filters
{
// std::map<std::string, hl_monitoring::Field::POIType> YoloV5OpenVino::stringToPOIEnum = {
//   { "ArenaCorner", hl_monitoring::Field::POIType::ArenaCorner },
//   { "LineCorner", hl_monitoring::Field::POIType::LineCorner },
//   { "T", hl_monitoring::Field::POIType::T },
//   { "X", hl_monitoring::Field::POIType::X },
//   { "Center", hl_monitoring::Field::POIType::Center },
//   { "PenaltyMark", hl_monitoring::Field::POIType::PenaltyMark },
//   { "PostBase", hl_monitoring::Field::POIType::PostBase }
// };

YoloV5OpenVino::YoloV5OpenVino() : Filter("YoloV5OpenVino"), model_path("model.pb")
{
  // TODO load classes from json config file

  // WARNING order is important (alphabetical order)
  //   if classes are removed, just comment the corresponding line, make sure to keep alphabetical order(!now Empty first)
  //classNames.push_back("ArenaCorner");
 // classNames.push_back("Empty");
//   classNames.push_back("Ball");
//   classNames.push_back("Center");
//   classNames.push_back("Empty");
//   classNames.push_back("LineCorner");
//   classNames.push_back("PenaltyMark");
//   classNames.push_back("PostBase");
//   classNames.push_back("Robot");
//   classNames.push_back("T");
//   classNames.push_back("X");
}

void YoloV5OpenVino::setParameters()
{
  debugLevel = ParamInt(0, 0, 1);
  confBallThreshold = ParamFloat(0.6, 0.0, 1.0);
  confPostThreshold = ParamFloat(0.6, 0.0, 1.0);
  confRobotThreshold = ParamFloat(0.6, 0.0, 1.0);
  nmsThreshold = ParamFloat(0.6, 0.0, 1.0);
  scoreThreshold = ParamFloat(0.8, 0.0, 1.0);
  //imSize = ParamInt(32, 1, 64);

  params()->define<ParamInt>("debugLevel", &debugLevel);
  params()->define<ParamFloat>("confBallThreshold", &confBallThreshold);
  params()->define<ParamFloat>("confPostThreshold", &confPostThreshold);
  params()->define<ParamFloat>("confRobotThreshold", &confRobotThreshold);
  params()->define<ParamFloat>("nmsThreshold", &nmsThreshold);
  params()->define<ParamFloat>("scoreThreshold", &scoreThreshold);
  //params()->define<ParamInt>("imSize", &imSize);

//   for (const std::string& className : classNames)
//   {
//     isUsingFeature[className] = ParamInt(0, 0, 1);
//     params()->define<ParamInt>("uses" + className, &(isUsingFeature[className]));
//   }
}

// void YoloV5OpenVino::updateUsedClasses()
// {
//   usedClassNames.clear();
//   for (size_t idx = 0; idx < classNames.size(); idx++)
//   {
//     const std::string& className = classNames[idx];
//     if (className == "Empty" || isUsingFeature[className] != 0)
//     {
//       usedClassNames.push_back(className);
//     }
//   }
// }

std::string YoloV5OpenVino::getClassName() const
{
  return "YoloV5OpenVino";
}

Json::Value YoloV5OpenVino::toJson() const
{
  Json::Value v = Filter::toJson();
  v["model_path"] = model_path;
  v["labels_path"] = labels_path;
  v["device"] = device;
  return v;
}

void YoloV5OpenVino::fromJson(const Json::Value& v, const std::string& dir_name)
{
  Filter::fromJson(v, dir_name);
  rhoban_utils::tryRead(v, "model_path", &model_path);
  rhoban_utils::tryRead(v, "labels_path", &labels_path);
  rhoban_utils::tryRead(v, "device", &device);

  init();
}

int YoloV5OpenVino::expectedDependencies() const
{
  return 1;
}

void YoloV5OpenVino::init()
{
    net = ie.ReadNetwork(model_path);
    inputs = net.getInputsInfo();
    outputs = net.getOutputsInfo();
    for (auto item : inputs)
    {
        inputsName = item.first;
        auto input_data = item.second;
        input_data->setPrecision(Precision::FP32);
        input_data->setLayout(Layout::NCHW);
        input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
        //std::cout << "input name = " << m_inputName << std::endl;
    }
 
 
    for (auto item : outputs)
    {
        auto output_data = item.second;
        output_data->setPrecision(Precision::FP32);
        outputsName = item.first;
        std::cout << "Loading model to the device " << device << std::endl;
    }
    auto executable_network = ie.LoadNetwork(net, device);
    infer_request = executable_network.CreateInferRequest();
 
    m_inputData = infer_request.GetBlob(inputsName);
    m_numChannels = m_inputData->getTensorDesc().getDims()[1];
    m_inputW = m_inputData->getTensorDesc().getDims()[2];
    m_inputH = m_inputData->getTensorDesc().getDims()[3];
    m_imageSize = m_inputH * m_inputW;

    net.setBatchSize(1);
    /** Reading labels (if specified) **/
    std::ifstream inputFile(labels_path);
    std::copy(std::istream_iterator<std::string>(inputFile),
                std::istream_iterator<std::string>(),
                std::back_inserter(labels));

    numClasses = labels.size();
}

double YoloV5OpenVino::sigmoid(double x){
    return (1 / (1 + exp(-x)));
}

std::vector<int> YoloV5OpenVino::getAnchors(int net_grid){
    std::vector<int> anchors(6);
    int a80[6] = {10,13, 16,30, 33,23};
    int a40[6] = {30,61, 62,45, 59,119};
    int a20[6] = {116,90, 156,198, 373,326}; 
    if(net_grid == 80){
        anchors.insert(anchors.begin(),a80,a80 + 6);
    }
    else if(net_grid == 40){
        anchors.insert(anchors.begin(),a40,a40 + 6);
    }
    else if(net_grid == 20){
        anchors.insert(anchors.begin(),a20,a20 + 6);
    }
    return anchors;
}

void YoloV5OpenVino::GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    // Generate index score pairs.
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }

    // Sort the score pair according to the scores in descending order
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend);

    // Keep top_k scores if needed.
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

void YoloV5OpenVino::NMSBoxes(const std::vector<cv::Rect>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold,
      std::vector<int>& indices,
      int limit = std::numeric_limits<int>::max())
{
    float eta = 1.f;
    int top_k = 0;
    // Get top_k scores (with corresponding indices).
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = rectOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep) {
            indices.push_back(idx);
            if (indices.size() >= limit) {
                break;
            }
        }
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}


void YoloV5OpenVino::detect(cv::Mat &image, std::vector<Detection> &outputs)
{
    cv::Mat blob_image;
    cv::resize(image, blob_image, cv::Size(m_inputW, m_inputH));
    cvtColor(blob_image, blob_image, cv::COLOR_BGR2RGB);
 
    float* data = static_cast<float*>(m_inputData->buffer());
    for (size_t row = 0; row < m_inputH; row++) {
        for (size_t col = 0; col < m_inputW; col++) {
            for (size_t ch = 0; ch < m_numChannels; ch++) {
#ifdef NCS2
				data[m_imageSize * ch + row * m_inputW + col] = float(blob_image.at<cv::Vec3b>(row, col)[ch]);
#else
				data[m_imageSize * ch + row * m_inputW + col] = float(blob_image.at<cv::Vec3b>(row, col)[ch] / 255.0);
#endif // NCS2
            }
        }
    }
    auto start = std::chrono::high_resolution_clock::now();
    infer_request.Infer();
    auto output = infer_request.GetBlob(outputsName);
	const float* detection_out = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());

    const InferenceEngine::SizeVector outputDims = output->getTensorDesc().getDims();
    float x_factor = (float)image.cols / (float)m_inputW;
    float y_factor = (float)image.rows / (float)m_inputH;
    float *dataout = (float *)detection_out;
    const int dimensions = outputDims[2];
    const int rows = outputDims[1];
    vector<int> class_ids;
    vector<float> confidences;
    vector<cv::Rect> boxes;
    for (int i = 0; i < rows; ++i)
    {
        float confidence = dataout[4];
        min_conf_threshold = std::min(std::min(static_cast<double>(confBallThreshold), static_cast<double>(confPostThreshold)), static_cast<double>(confRobotThreshold));
        if (confidence >= min_conf_threshold)
        {
            float * classes_scores = dataout + 5;
            cv::Mat scores(1, labels.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > scoreThreshold)
            {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
 
                float x = dataout[0];
                float y = dataout[1];
                float w = dataout[2];
                float h = dataout[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        dataout += dimensions;
    }
    vector<int> nms_result;
    NMSBoxes(boxes, confidences, min_conf_threshold, nmsThreshold, nms_result);
    for (int i = 0; i < nms_result.size(); i++)
    {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.label = labels[result.class_id];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
		outputs.push_back(result);
    }
}

void YoloV5OpenVino::process()
{
    clearAllFeatures();
    frame = (*(getDependency().getImg())).clone();
    std::vector<Detection> objects;
    detect(frame, objects);
    for (auto &object : objects) {
            /** Drawing only objects when >confidence_threshold probability **/
        draw_result = false;
        if (object.label == "ball") {
            if (object.confidence >= confBallThreshold) {
                cv::Point2f object_center(object.box.x + object.box.width/2, object.box.y + object.box.height/2);
                pushBall(object_center);
                draw_result = true;
            }
        }
        else if (object.label == "robot") {
            if (object.confidence >= confRobotThreshold) {
                cv::Point2f object_botom_center(object.box.x + object.box.width/2, object.box.y + object.box.height);
                pushRobot(object_botom_center);
                draw_result = true;
            }
        }
        else if (object.label == "goalpost") {
            if (object.confidence >= confPostThreshold) {
                cv::Point2f object_botom_center(object.box.x + object.box.width/2, object.box.y + object.box.height);
                pushPOI(hl_monitoring::Field::POIType::PostBase, object_botom_center);
                draw_result = true;
            }
        }
        else
            logger.error("Unknown label: %s", object.label);
        
        if (draw_result){
            cv::putText(frame,
                (object.class_id < static_cast<int>(labels.size()) ?
                        object.label : std::string("label #") + std::to_string(object.class_id)) + std::string(" ") + std::to_string(object.confidence),
                    cv::Point2f(static_cast<float>(object.box.x), static_cast<float>(object.box.y - 5)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                    cv::Scalar(250, 250, 0));
            cv::rectangle(frame, object.box, cv::Scalar(250, 250, 0));
        }
        
    }
    //std::cerr << "RETURNING IMAGE " << std::endl;
    cv::resize(frame, frame, cv::Size(720,540));
    img() = frame;
    
    
    // if (isModeChanged) {
    //     isModeChanged = false;
    // }

    // // Final point:
    // // in the truly Async mode, we swap the NEXT and CURRENT requests for the next iteration
    // frame = next_frame;
    // next_frame = cv::Mat();
    // if (isAsyncMode) {
    //     async_infer_request_curr.swap(async_infer_request_next);
    // }

}
}  // namespace Filters
}  // namespace Vision

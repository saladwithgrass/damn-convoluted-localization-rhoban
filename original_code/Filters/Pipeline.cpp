#include <stdexcept>
#include <list>
#include <functional>
#include <fstream>
#include "Filters/Source/Source.hpp"
#include "Filters/Pipeline.hpp"
#include <iostream>
#include "Filters/FilterFactory.hpp"
#include "CameraState/CameraState.hpp"

#include "rhoban_utils/timing/benchmark.h"
#include "rhoban_utils/logging/logger.h"

#include <future>

// Divider at which the stereo processing is done in respect to classic pipeline FPS.
// Set to 1 to perform stereoProcessing each captured frame, 2 for each second frame etc.
// Look for a warnings about "stereoimgProc results not ready when being requested" to estimate the right value
// TODO: maybe rhiorize this?
#define STEREO_DIVIDER 2

#define PRINT_PIPELINE_TOPOLOGY 0

// Enable this to run filters in parallel batches
// But tests shows almost no speedup when most of the filters are already parallelised by opencv parallel_for_, etc
// Warning: benchmark will show all paralellised filters as "unknown" in it's report
#define USE_ASYNC_BATCHES 1

// Number of worker threads for parallel batches
#define ASYNC_THREADS_NUMBER 8

using Vision::Utils::CameraState;

using namespace std;
using namespace rhoban_utils;



double pipeline_start;

namespace Vision {

// Static variables
bool Pipeline::stereoSourceReady;
bool Pipeline::stereoProcessed;

rhoban_utils::Logger logger("Pipeline");

// Declatin a thread pool of parallel workers for pipeline filters
ThreadPool Pipeline::threadPool(ASYNC_THREADS_NUMBER);

Pipeline::Pipeline() : _filters(), _rootFilters(), _children(), _filterThread(), _timestamp() {
  cs = NULL;
  cs_slow_stereo = NULL;

  stopStereoThread = false;

  stereoSourceReady = false;
  stereoProcessed = false;

  first_step = true;

  pipline_start_ts = TimeStamp::now();
}

Pipeline::~Pipeline() {
  // Free Filters
  _rootFilters.clear();
  _children.clear();
  for (auto& it : _filters) {
    delete it.second;
  }
  _filters.clear();

  stopStereoThread = true;
  if (_stereoThread != NULL) {
    _stereoThread->join();
    delete _stereoThread;
  }
}

Utils::CameraState* Pipeline::getCameraState(bool isSlowStereo) {
  if(isSlowStereo) return cs_slow_stereo;
  return cs; 
}

void Pipeline::setCameraState(Utils::CameraState* csInit, bool isSlowStereo) { 
  if(isSlowStereo) cs_slow_stereo = csInit;
  else cs = csInit; 
}

void Pipeline::add(Filter* filter) {
  if (filter == nullptr) {
    throw std::logic_error("Pipeline null filter");
  }
  if (_filters.count(filter->getName()) != 0) {
    throw std::logic_error("Pipeline filter name already register: '" + filter->getName() + "'");
  }

  _filters[filter->getName()] = filter;
  filter->_pipeline = this;

  if (filter->_dependencies.size() == 0) {
    _rootFilters.push_back(filter);
  }
}

void Pipeline::add(std::vector<std::unique_ptr<Filter>>* filters) {
  for (size_t idx = 0; idx < filters->size(); idx++) {
    add((*filters)[idx].release());
  }
}

const Filter& Pipeline::get(const std::string& name) const {
  try {
    return *(_filters.at(name));
  } catch (const std::out_of_range& e) {
    throw std::runtime_error("No filters named '" + name + "' found in pipeline");
  }
}
Filter& Pipeline::get(const std::string& name) {
  try {
    return *(_filters.at(name));
  } catch (const std::out_of_range& e) {
    cout << "No filters named '" + name + "' found in pipeline" << endl;
    throw std::runtime_error("No filters named '" + name + "' found in pipeline");
  }
}

bool Pipeline::isFilterPresent(const std::string& name) { return _filters.find(name) != _filters.end(); }

const Pipeline::FiltersMap& Pipeline::filters() const { return _filters; }
Pipeline::FiltersMap& Pipeline::filters() { return _filters; }

void Pipeline::step(Filter::UpdateType updateType) {
  if (first_step) {
    first_step = false;
    if (isFilterPresent("stereoImgProc")) {
      std::cout << "---- stereoImgProc filter found, Stereo mode ENABLED ---- " << std::endl;
      _stereoThread = new std::thread(std::bind(&Pipeline::stereoThreadMainFunc, this));
      stereoImgProcFilter = _filters["stereoImgProc"];
      stereo_divider = 0;
    } else {
      std::cout << "---- stereoImgProc filter not found, Stereo mode DISABLED ---- " << std::endl;
      stereoSourceReady = true;
      stereoProcessed = true;
      stopStereoThread = true;
    }
  }

  pipeline_start = (double)cv::getTickCount();

  Benchmark::open("Resolve dependencies");
  resolveDependencies();
  Benchmark::close("Resolve dependencies");

  // Apply previous on nonDependency filters
  std::list<Filter*> list;
  std::map<std::string, int> dependenciesSolved;

  double stereoDeltaMs = 0;

  Filter* first_filter;
  Filter* second_filter;

  double interFrameTimeMs = 1000;
  for (Filter* filter : _rootFilters) {
    Filters::Source* src = dynamic_cast<Filters::Source*>(filter);
    if (src != nullptr) {
      double framerate = src->getFrameRate();
      if (framerate) interFrameTimeMs = 1000.0 / framerate;
    }
  }

  do {
    bool first_timestamp_received = false;
    TimeStamp first_frame_ts;
    // Vector of async threads for source filters
    std::vector<std::future<void>> results_src;

    // filter->runStep(updateType);

    // Starting all source filters processing in separate threads
    if (fabs(stereoDeltaMs) <= interFrameTimeMs / 2) {
      // Need to capture both
      for (Filter* filter : _rootFilters) {
#if PRINT_PIPELINE_TOPOLOGY
        std::cout << "Processing source: " << filter->getName() << std::endl;
#endif

        results_src.emplace_back(std::async(std::launch::async, &Filter::runStep, filter, updateType));
      }
    } else if (stereoDeltaMs < 0) {
      // need to recapture first
      results_src.emplace_back(std::async(std::launch::async, &Filter::runStep, first_filter, updateType));
    } else {
      // need to recapture second
      results_src.emplace_back(std::async(std::launch::async, &Filter::runStep, second_filter, updateType));
    }

    // Waiting for all source filters to finish
    for (auto&& result : results_src) result.wait();

    // Checking source filter timestamps and generating cameraState
    for (Filter* filter : _rootFilters) {
      Filters::Source* src = dynamic_cast<Filters::Source*>(filter);
      if (src != nullptr) {
        TimeStamp this_frame_ts = src->frame_ts;
        if (!first_timestamp_received) {
          first_frame_ts = this_frame_ts;
          first_filter = filter;
          first_timestamp_received = true;
        } else {
          second_filter = filter;
          stereoDeltaMs = diffMs(this_frame_ts, first_frame_ts);
          std::cout << "stereoDeltaMs=" << stereoDeltaMs << ", first=" << diffMs(pipline_start_ts, first_frame_ts)
                    << ", this=" << diffMs(pipline_start_ts, this_frame_ts) << std::endl;
          // if stereoDeltaMs is negative - it means that this_frame_ts > first_frame_ts
          // TODO: stereoDeltaMs will slowly raise with time because of differents in clock crystals in cams
          // Tests shows ~5-7ms drift in 10 minutes on spinnaker cameras. Need to be fixed
        }
      }
    }
  } while (fabs(stereoDeltaMs) > interFrameTimeMs / 2);

  for (Filter* filter : _rootFilters) {
    Filters::Source* src = dynamic_cast<Filters::Source*>(filter);
    if (src != nullptr) {
      // Vital step : after processing the source filter, the timestamp of the
      // image is known
      // and the cameraState needs to be updated.
      frames++;

      switch (src->getType()) {
        case Filters::Source::Log:
        case Filters::Source::Online: {
          //printf("Filters::Source::Online\r\n");
          //sourceRaw/sourceRaw2 is "Online" filters
          if (cs != nullptr) {
            cs->setClockOffset(src->getClockOffset());
            //printf("Calling updateInternalModel with ts=%f\r\n", _timestamp.getTimeSec());
            cs->updateInternalModel(_timestamp);
          }
          if(stereo_divider == 0) {
            if (cs_slow_stereo != nullptr) {
              cs_slow_stereo->setClockOffset(src->getClockOffset());
              cs_slow_stereo->updateInternalModel(_timestamp);
            }          
          }
          break;
        }
        case Filters::Source::Custom:
          //printf("Filters::Source::Custom\r\n");
          if (cs != nullptr) {
            delete (cs);
          }
          setCameraState(src->buildCameraState());
          // cs->setClockOffset(src->getClockOffset());
          break;
      }
    }
    for (auto& son : _children.at(filter->getName())) {
      std::string sonName = son->getName();

      dependenciesSolved[sonName]++;
      if ((size_t)dependenciesSolved[sonName] == _filters[sonName]->_dependencies.size()) {
        list.push_back(son);
#if PRINT_PIPELINE_TOPOLOGY
        std::cout << "   Ready to process: " << sonName << " ( ";  // << std::endl;
        Filter::Dependencies* deps = &_filters[sonName]->_dependencies;
        for (size_t i = 0; i < deps->size(); i++) {
          std::cout << (*deps)[i] << " ";
        }
        std::cout << ")" << std::endl;
#endif
      }
    }
  }

  // Sweep through topological order
  // Do no process a filter twice
  while (!list.empty()) {
    size_t popped_count = 0;
    std::vector<Filter*> popped_filters;
    ;

    while (!list.empty() && popped_count < ASYNC_THREADS_NUMBER) {
      Filter* filter = list.front();
      list.pop_front();
      popped_filters.emplace_back(filter);
      popped_count++;
    }

    // Batch is a filter group which can be run in parallel because all of them are dependent from already satisfied
    // resources i.e. YWideangle, ballRadiusProviderWideangle, birdview, greenHSVWideangle, source - all depends from
    // sourceRaw and can be run in parallel after sourceRaw is ready
    std::stringstream batch_ss;
    batch_ss << "Filters batch: ";
    for (size_t i = 0; i < popped_count; i++) batch_ss << popped_filters[i]->getName() << ", ";
    // std::cout << std::endl;

    double batch_start_timer_ms = 1000.0 * ((double)cv::getTickCount()) / (double)cv::getTickFrequency();

    std::vector<std::future<void>> results;

    #if USE_ASYNC_BATCHES
    Benchmark::open(batch_ss.str());
    #endif

    for (size_t i = 0; i < popped_count; i++) {
      Filter* filter = popped_filters[i];
      if (filter->getName() == "stereoImgProc") {  // TODO: dirty hardcode
        if(stereo_divider==0) {
          // Notifying stereoThread that source pictures are ready
          {
            // Here and below: unique_lock is a smart-pointer-like hack.
            // It will lock mutex on creation and unlock automatically when unique_lock goues out of scope
            // That's why we are using additional brackets for this code snippet
            std::unique_lock<std::mutex> lk(steroProcessingMutex);
            stereoSourceReady = true;
            stereoProcessed = false;
          }
          cv.notify_all();
        }
      } else {
        #if PRINT_PIPELINE_TOPOLOGY
        std::cout << "Processing: " << filter->getName() << std::endl;
        #endif
        #if USE_ASYNC_BATCHES

        // Uncomment this to use std::async as async engine
        results.emplace_back(std::async(std::launch::async, &Filter::runStep, filter, Filter::forward));

        // Uncoment this to use ThreadPool as async engine (took from https://github.com/progschj/ThreadPool)
        // results.emplace_back( threadPool.enqueue(&Filter::runStep, filter, Filter::forward) );

        #else
        double filter_start_timer_ms = 1000.0 * ((double)cv::getTickCount()) / (double)cv::getTickFrequency();
        filter->runStep();
        double filter_stop_timer_ms = 1000.0 * ((double)cv::getTickCount()) / (double)cv::getTickFrequency();
        std::cout << "    Filter " << filter->getName() << " time : " << filter_stop_timer_ms - filter_start_timer_ms
                  << " ms" << std::endl;
        #endif
      }
    }

#if USE_ASYNC_BATCHES
    // Waiting for all filters in batch to finish their jobs
    for (auto&& result : results) result.wait();
    Benchmark::close(batch_ss.str().c_str());
#endif

    double batch_stop_timer_ms = 1000.0 * ((double)cv::getTickCount()) / (double)cv::getTickFrequency();

    // std::cout << "BATCH TIME: " << batch_stop_timer_ms - batch_start_timer_ms << " ms" << std::endl;

    // Preparing the list of filters that can be run in next batch using resources computated above
    for (size_t i = 0; i < popped_count; i++) {
      Filter* filter = popped_filters[i];

      for (auto& son : _children.at(filter->getName())) {
        std::string sonName = son->getName();
        dependenciesSolved[sonName]++;
        if ((size_t)dependenciesSolved[sonName] == _filters[sonName]->_dependencies.size()) {
          list.push_back(son);
#if PRINT_PIPELINE_TOPOLOGY
          std::cout << "   Ready to process: " << sonName << " ( ";  // << std::endl;
          Filter::Dependencies* deps = &_filters[sonName]->_dependencies;
          for (size_t i = 0; i < deps->size(); i++) {
            std::cout << (*deps)[i] << " ";
          }
          std::cout << ")" << std::endl;
#endif
        }
      }
    }
    // list.pop_front();
  }

  double pipeline_before_stereo_time_ms =
      1000.0 * ((double)cv::getTickCount() - pipeline_start) / (double)cv::getTickFrequency();
  // std::cout << "PIPELINE TIME BEFORE STEREO: " << pipeline_before_stereo_time_ms << " ms" << std::endl;

  stereo_divider++;
  if(stereo_divider == STEREO_DIVIDER) {
    // Classic pipeline processed STEREO_DIVIDER times. It's time to get results from stereoimgProc filter
    stereo_divider = 0;
    if(!stereoProcessed) {
      logger.warning("stereoimgProc results not ready when being requested in pipeline, please adjust STEREO_DIVIDER");
    }
    // Waiting for stereo image to be processed before switching to next pipeline step
    {
      std::unique_lock<std::mutex> lk(steroProcessingMutex);
      while (!stereoProcessed) {
        cv.wait(lk);
      }
    }
  }

  double pipeline_with_stereo_time_ms =
      1000.0 * ((double)cv::getTickCount() - pipeline_start) / (double)cv::getTickFrequency();
  // std::cout << "PIPELINE TIME WITH STEREO: " << pipeline_with_stereo_time_ms << " ms" << std::endl;

}

void Pipeline::runStep() { step(Filter::UpdateType::forward); }

void Pipeline::finish() {
  for (auto& entry : filters()) {
    entry.second->finish();
  }
}

void Pipeline::resolveDependencies() {
  if (_children.size() != 0) {
    return;
  }

  for (auto& it : _filters) {
    _children[it.first] = std::vector<Filter*>();
  }

  // Preparing children map
  for (auto& it : _filters) {
    for (auto& dep : it.second->_dependencies) {
      _children[dep].push_back(it.second);
    }
  }

  // Checking consistency, is there
  for (auto& pair : _children) {
    const std::string& father = pair.first;
    try {
      _filters.at(father);
    } catch (const std::out_of_range& exc) {
      std::ostringstream oss;
      oss << "Unknown dependency '" << father << "', required from the following filters: ( ";
      for (size_t index = 0; index < pair.second.size(); index++) {
        oss << "'" << pair.second[index]->getName() << "'";
        if (index != pair.second.size() - 1) oss << ",";
      }
      oss << ")";
      throw std::runtime_error(oss.str());
    }
  }

  //[Sol] print pipeline structure
  std::cout << "Pipeline contents:" << std::endl;
  for (auto& pair : _children) {
    const std::string& name = pair.first;
    std::cout << name << " ( ";

    Filter::Dependencies* deps = &_filters.at(name)->_dependencies;
    for (size_t i = 0; i < deps->size(); i++) {
      std::cout << (*deps)[i] << " ";
    }
    std::cout << " )" << std::endl;
  }
}

Json::Value Pipeline::toJson() const {
  Json::Value v;
  for (auto& f : _filters) {
    v.append(f.second->toJson());
  }
  return v;
}

void Pipeline::addFiltersFromJson(const Json::Value& v, const std::string& dir_name) {
  Filters::FilterFactory ff;
  std::vector<std::unique_ptr<Filter>> result;
  if (v.isArray()) {
    for (Json::ArrayIndex idx = 0; idx < v.size(); idx++) {
      add(ff.build(v[idx], dir_name).release());
    }
  } else if (v.isObject()) {
    if (!v.isMember("filters") && !v.isMember("paths")) {
      throw JsonParsingError(DEBUG_INFO + " pipeline should contain either 'filters' or 'paths'");
    }
    if (v.isMember("filters")) {
      if (!v["filters"].isArray()) {
        throw JsonParsingError(DEBUG_INFO + " expecting an array for 'filters'");
      }
      try {
        addFiltersFromJson(v["filters"], dir_name);
      } catch (const JsonParsingError& err) {
        throw JsonParsingError(std::string(err.what()) + " in 'filters'");
      }
    }
    std::vector<std::string> paths;
    rhoban_utils::tryReadVector(v, "paths", &paths);
    for (const std::string& path : paths) {
      std::string file_path = dir_name + path;
      std::string read_dir_path = rhoban_utils::getDirName(file_path);
      Json::Value path_value = file2Json(file_path);
      try {
        std::cout << "adding from " << file_path << std::endl;
        addFiltersFromJson(path_value, read_dir_path);
      } catch (const JsonParsingError& err) {
        throw JsonParsingError(std::string(err.what()) + " in '" + file_path + "'");
      }
    }
  } else {
    throw rhoban_utils::JsonParsingError(DEBUG_INFO + " pipeline is not an array neither an object");
  }
}

void Pipeline::fromJson(const Json::Value& v, const std::string& dir_name) {
  addFiltersFromJson(v, dir_name);
  std::cout << "There is now " << _filters.size() << " filters." << std::endl;

  if (v.isObject() && v.isMember("default_camera_state")) {
    cs = new Utils::CameraState(nullptr);
    cs->_cameraModel.read(v["default_camera_state"], "camera_model", dir_name);
    cs_slow_stereo = new Utils::CameraState(nullptr);
    cs_slow_stereo->_cameraModel.read(v["default_camera_state"], "camera_model", dir_name);    
  }
}

void Pipeline::setTimestamp(const ::rhoban_utils::TimeStamp& ts) {
  //printf("Calling Pipeline::setTimestamp with ts=%f\r\n", ts.getTimeSec());
  _timestamp = ts;
}

const rhoban_utils::TimeStamp& Pipeline::getTimestamp() const { return _timestamp; }

void Pipeline::hideAllFilters() {
  for (auto& it : _filters) it.second->display = false;
}

void Pipeline::stereoThreadMainFunc() {
  while (!stopStereoThread) {
    // Waiting for stereo source images to be ready
    {
      std::unique_lock<std::mutex> lk(steroProcessingMutex);
      while (!stereoSourceReady) {
        cv.wait(lk);
      }
    }
    // Doing stereo processing
    Benchmark::open("Stereo processing in separate thread");  // This will be a root benchmark for this thread
    stereoImgProcFilter->runStep(Filter::forward);
    Benchmark::close("Stereo processing in separate thread", benchmarkFromRobocup, benchmarkDetailFromRobocup);

    // Stereo processing done, notifying main thread about it
    {
      std::unique_lock<std::mutex> lk(steroProcessingMutex);
      stereoProcessed = true;
      stereoSourceReady = false;
    }
    cv.notify_all();
  }
}

}  // namespace Vision

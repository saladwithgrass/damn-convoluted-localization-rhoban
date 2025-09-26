#include "SourceFactory.hpp"

#include "SourceLogs.hpp"
#include "SourceOpenCV.hpp"
#include "SourceVideoProtobuf.hpp"
#include "SourceSpinnaker.hpp"
#include "SourceROS.hpp"
#include "SourceWebots.hpp"

#include "../FilterFactory.hpp"

#ifdef KID_SIZE_USES_FLYCAPTURE
#include "SourcePtGrey.hpp"
#endif
#ifdef KID_SIZE_USES_IDS
#include "SourceIDS.hpp"
#endif

namespace Vision {
namespace Filters {
void registerSourceFilters(FilterFactory* ff) {
  ff->registerBuilder("SourceLogs", []() { return std::unique_ptr<Filter>(new SourceLogs); });
  ff->registerBuilder("SourceOpenCV", []() { return std::unique_ptr<Filter>(new SourceOpenCV); });
  ff->registerBuilder("SourceVideoProtobuf", []() { return std::unique_ptr<Filter>(new SourceVideoProtobuf); });
#ifdef KID_SIZE_USES_FLYCAPTURE
  ff->registerBuilder("SourcePtGrey", []() { return std::unique_ptr<Filter>(new SourcePtGrey); });
#endif
#ifdef KID_SIZE_USES_IDS
  ff->registerBuilder("SourceIDS", []() { return std::unique_ptr<Filter>(new SourceIDS); });
#endif
  ff->registerBuilder("SourceSpinnaker", []() { return std::unique_ptr<Filter>(new SourceSpinnaker); });
  ff->registerBuilder("SourceROS", []() { return std::unique_ptr<Filter>(new SourceROS); });
  ff->registerBuilder("SourceWebots", []() { return std::unique_ptr<Filter>(new SourceWebots); });
}

}  // namespace Filters
}  // namespace Vision

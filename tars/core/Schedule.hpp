#ifndef TARS_CORE_SCHEDULE_H_
#define TARS_CORE_SCHEDULE_H_

#include <stdio.h>

#include <array>
#include <map>
#include <string>
#include <vector>

#include "core/Command.hpp"
#include "core/Interpreter.hpp"
#include "core/device.h"
namespace tars {

struct Op;
struct Net;

// net scheduler
class Schedule {
 public:
  enum Type {
    // Size can be compute separately
    SEPARATE = 0,
    // When size is fixed, the content is fixed
    CONSTANT = 1,
    // Size can't be compute separately
    NOT_SEPERATE
  };

  // pipeline info
  struct OpCacheInfo {
    // op
    const Op* op;
    // input tensors
    std::vector<Tensor*> inputs;
    // output tensors
    std::vector<Tensor*> outputs;
    // schedule type
    Schedule::Type type = Schedule::Type::SEPARATE;

    // Command buffer for cache
    CommandBuffer cacheBuffer;

    // Command buffer for execute
    CommandBuffer executeBuffer;

    std::map<const Op*, std::shared_ptr<Operation>> executionCache;
  };

  // Device, Tensor, shape-dirty, content-dirty
  typedef std::tuple<Tensor*, std::shared_ptr<Tensor>, bool, bool> TENSORCACHE;

  struct BackendCache {
    Device::Info info;
    BackendConfig config;
    std::pair<std::shared_ptr<Device>, std::shared_ptr<Device>> cache;
    bool needComputeShape = true;
    bool needComputeGeometry = true;
    std::map<Tensor*, TENSORCACHE> inputTensorCopyCache;
  };

  typedef std::pair<BackendCache, std::vector<OpCacheInfo>> PipelineInfo;

  // schedule info
  struct ScheduleInfo {
    // pipelines with backend info
    std::vector<PipelineInfo> pipelineInfo;
    // input tensors map
    std::map<std::string, Tensor*> inputTensors;
    // output tensors map
    std::map<std::string, Tensor*> outputTensor;
    // all tensors
    std::vector<std::shared_ptr<Tensor>> allTensors;
    // input valid for resize
    bool validForResize;
    // Default Device
    std::shared_ptr<Device> defaultBackend;
    // size need input's content
    bool needInputContentForShape = false;
  };

  // schedule net ops to pipeline with configuration.
  static bool schedule(ScheduleInfo& result, const Net* net,
                       const std::vector<ScheduleConfig>& config,
                       const RuntimeInfo& runtimeInfo);

  static MNNForwardType getApprociateType(const ScheduleConfig& config);
};
}  // namespace tars

#endif

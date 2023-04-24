#ifndef Schedule_hpp
#define Schedule_hpp

#include <stdio.h>

#include <ace/Interpreter.hpp>
#include <array>
#include <map>
#include <string>
#include <vector>

#include "core/Backend.hpp"
#include "core/TensorUtils.hpp"

namespace ace {

struct Op;
struct Net;

/** net scheduler */
class Schedule {
 public:
  enum Type {
    // Size can be compute seperately
    SEPERATE = 0,
    // When size is fixed, the content is fixed
    CONSTANT = 1,
    // Size can't be compute seperately
    NOT_SEPERATE
  };

  struct PipelineInfo {
    const Op* op;
    // input tensors
    std::vector<Tensor*> inputs;
    // output tensors
    std::vector<Tensor*> outputs;
    // schedule type
    Schedule::Type type = Schedule::Type::SEPERATE;
  };

  struct ScheduleInfo {
    // pipelines with backend info
    std::vector<std::pair<Backend::Info, std::vector<PipelineInfo>>>
        pipelineInfo;
    // input tensors map
    std::map<std::string, Tensor*> inputTensors;
    // output tensors map
    std::map<std::string, Tensor*> outputTensor;
    // all tensors map
    std::vector<std::pair<int, std::shared_ptr<Tensor>>> allTensors;
    // input valid for resize
    bool validForResize;
  };

  // schedule net ops to pipeline with configuration.
  static ScheduleInfo schedule(const Net* net,
                               const std::vector<ScheduleConfig>& config);
  static DeviceType getApprociateType(const ScheduleConfig& config);
};
}  // namespace ace

#endif /* Schedule_hpp */

//
//  CPURange.cpp
//  MNN
//
//  Created by MNN on 2018/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "device/cpu/CPUDevice.h"
#include "device/cpu/CPURange.hpp"

namespace tars {

template <typename T>
CPURange<T>::CPURange(Device* backend) : Operation(backend) {
  // nothing to do
}

template <typename T>
Status CPURange<T>::onExecute(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs) {
  const T start = inputs[0]->host<T>()[0];
  const T delta = inputs[2]->host<T>()[0];
  int32_t outputSize = outputs[0]->buffer().dim[0].extent;
  auto flat = outputs[0]->host<T>();
  T val = start;
  for (int32_t i = 0; i < outputSize; ++i) {
    flat[i] = T(val);
    val += delta;
  }
  return Status::OK();
}

class CPURangeCreator : public CPUDevice::Creator {
 public:
  virtual Operation* onCreate(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs,
                              const tars::Op* op, Device* backend) const {
    auto code = inputs[0]->getType().code;
    switch (code) {
      case halide_type_int:
        return new tars::CPURange<int32_t>(backend);
      case halide_type_float:
        return new tars::CPURange<float>(backend);
      default:
        MNN_ASSERT(false);  // unsupported type
        return nullptr;
    }
  }
};

REGISTER_CPU_OP_CREATOR(CPURangeCreator, OpType_Range);
}  // namespace tars

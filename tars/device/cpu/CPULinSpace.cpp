//
//  CPULinSpace.cpp
//  MNN
//
//  Created by MNN on 2019/12/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "device/cpu/CPUDevice.h"
#include "device/cpu/CPULinSpace.hpp"

namespace tars {
Status CPULinSpace::onExecute(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs) {
  MNN_ASSERT(inputs.size() == 3);
  MNN_ASSERT(outputs.size() == 1);
  const float start = inputs[0]->host<float>()[0];
  const float stop = inputs[1]->host<float>()[0];
  const int num = inputs[2]->host<int32_t>()[0];
  MNN_ASSERT(num > 0);

  float* outputData = outputs[0]->host<float>();

  if (num == 1) {
    outputData[0] = start;
    return Status::OK();
  }

  if (num == 2) {
    outputData[0] = start;
    outputData[1] = stop;
    return Status::OK();
  }

  // make sure that start with the first and end with the last.
  outputData[0] = start;
  outputData[num - 1] = stop;
  const float step = (stop - start) / (num - 1);
  for (int i = 1; i < num - 1; ++i) {
    outputData[i] = start + i * step;
  }

  return Status::OK();
}

class CPULinSpaceCreator : public CPUDevice::Creator {
 public:
  virtual Operation* onCreate(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs,
                              const tars::Op* op, Device* backend) const {
    return new CPULinSpace(backend);
  }
};

REGISTER_CPU_OP_CREATOR(CPULinSpaceCreator, OpType_LinSpace);
}  // namespace tars

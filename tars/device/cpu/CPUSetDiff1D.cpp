//
//  CPUSetDiff1D.cpp
//  MNN
//
//  Created by MNN on 2019/6/11.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#include "device/cpu/CPUSetDiff1D.hpp"
namespace tars {
Status CPUSetDiff1D::onExecute(const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) {
  auto input = inputs[0];
  auto remove = inputs[1];
  if (input->getType().code != halide_type_int ||
      remove->getType().code != halide_type_int) {
    return Status::ERROR();
  }
  auto output = outputs[0];
  auto outputPtr = output->host<int32_t>();
  auto inputPtr = input->host<int32_t>();
  auto removePtr = remove->host<int32_t>();
  auto removeSize = remove->elementSize();
  auto inputSize = input->elementSize();
  int outputSize = 0;
  for (int i = 0; i < inputSize; ++i) {
    auto value = inputPtr[i];
    bool valid = true;
    for (int j = 0; j < removeSize; ++j) {
      if (value == removePtr[j]) {
        valid = false;
        break;
      }
    }
    if (valid) {
      outputPtr[outputSize++] = value;
    }
  }
  output->setLength(0, outputSize);
  return Status::OK();
}
class CPUSetDiff1DCreator : public CPUDevice::Creator {
 public:
  virtual Operation *onCreate(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs,
                              const tars::Op *op, Device *backend) const {
    return new CPUSetDiff1D(backend);
  }
};

REGISTER_CPU_OP_CREATOR(CPUSetDiff1DCreator, OpType_SetDiff1D);

};  // namespace tars

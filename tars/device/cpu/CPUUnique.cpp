//
//  CPUUnique.cpp
//  MNN
//
//  Created by MNN on 2019/6/11.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#include <unordered_map>

#include "device/cpu/CPUUnique.hpp"
namespace tars {

Status CPUUnique::onExecute(const std::vector<Tensor *> &inputs,
                            const std::vector<Tensor *> &outputs) {
  auto input = inputs[0];
  if (input->getType().code != halide_type_int) {
    return Status::ERROR();
  }
  auto output = outputs[0];
  auto outputPtr = output->host<int32_t>();
  int outputSize = 0;
  std::unordered_map<int, int> idx_map;
  auto eleSize = input->elementSize();
  for (int i = 0; i < eleSize; ++i) {
    auto value = input->host<int32_t>()[i];
    if (idx_map.find(value) == idx_map.end()) {
      outputPtr[outputSize] = value;
      idx_map[value] = outputSize++;
    }
  }
  if (outputs.size() > 1) {
    auto outIdx = outputs[1]->host<int>();
    for (int i = 0; i < eleSize; ++i) {
      auto value = input->host<int32_t>()[i];
      outIdx[i] = idx_map[value];
    }
  }
  return Status::OK();
}
class CPUUniqueCreator : public CPUDevice::Creator {
 public:
  virtual Operation *onCreate(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs,
                              const tars::Op *op, Device *backend) const {
    return new CPUUnique(backend);
  }
};

REGISTER_CPU_OP_CREATOR(CPUUniqueCreator, OpType_Unique);

};  // namespace tars

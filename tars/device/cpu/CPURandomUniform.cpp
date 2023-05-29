//
//  CPURandomUniform.cpp
//  MNN
//
//  Created by MNN on 2020/8/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <random>

#include "core/Macro.h"
#include "device/cpu/CPUDevice.h"
#include "device/cpu/CPURandomUniform.hpp"

namespace tars {
Status CPURandomUniform::onResize(const std::vector<Tensor*>& inputs,
                                  const std::vector<Tensor*>& outputs) {
  return Status::OK();
}

Status CPURandomUniform::onExecute(const std::vector<Tensor*>& inputs,
                                   const std::vector<Tensor*>& outputs) {
  MNN_ASSERT(outputs.size() == 1);
  auto output = outputs[0];
  int size = output->elementSize();
  auto parameter = mOp->main_as_RandomUniform();
  auto outputPtr = output->host<float>();
  std::uniform_real_distribution<float> distribution(parameter->low(),
                                                     parameter->high());
  int seed = parameter->seed();
  int seed1 = parameter->seed2();
  if (seed || seed1) {
    std::mt19937 generator(seed || seed1);
    for (int i = 0; i < size; i++) {
      outputPtr[i] = distribution(generator);
    }
  } else {
    std::default_random_engine generator;
    for (int i = 0; i < size; i++) {
      outputPtr[i] = distribution(generator);
    }
  }
  return Status::OK();
}

Status CPURandomNormal::onResize(const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) {
  return Status::OK();
}

Status CPURandomNormal::onExecute(const std::vector<Tensor*>& inputs,
                                  const std::vector<Tensor*>& outputs) {
  MNN_ASSERT(outputs.size() == 1);
  auto output = outputs[0];
  int size = output->elementSize();
  auto parameter = mOp->main_as_RandomUniform();
  auto outputPtr = output->host<float>();
  // RandomUniform and RandomNormal use same param table. low -> mean, high ->
  // scale
  std::normal_distribution<float> distribution(parameter->low(),
                                               parameter->high());
  int seed = parameter->seed();
  int seed1 = parameter->seed2();
  if (seed || seed1) {
    std::mt19937 generator(seed || seed1);
    for (int i = 0; i < size; i++) {
      outputPtr[i] = distribution(generator);
    }
  } else {
    std::default_random_engine generator;
    for (int i = 0; i < size; i++) {
      outputPtr[i] = distribution(generator);
    }
  }
  return Status::OK();
}

class CPURandomCreator : public CPUDevice::Creator {
 public:
  virtual Operation* onCreate(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs,
                              const tars::Op* op,
                              Device* backend) const override {
    if (op->type() == OpType_RandomUniform) {
      return new CPURandomUniform(backend, op);
    } else {
      return new CPURandomNormal(backend, op);
    }
  }
};
REGISTER_CPU_OP_CREATOR(CPURandomCreator, OpType_RandomUniform);
REGISTER_CPU_OP_CREATOR(CPURandomCreator, OpType_RandomNormal);
}  // namespace tars

//
//  CPUUnravelIndex.cpp
//  MNN
//
//  Created by MNN on 2019/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/OpCommonUtils.hpp"
#include "device/cpu/CPUDevice.h"
#include "device/cpu/CPUUnravelIndex.hpp"

namespace tars {

Status CPUUnravelIndex::onExecute(const std::vector<Tensor*>& inputs,
                                  const std::vector<Tensor*>& outputs) {
  auto indices = inputs[0];
  auto dims = inputs[1];

  const int elmentSize = indices->elementSize();
  const int dimsSize = dims->length(0);

  const auto indicesPtr = indices->host<int32_t>();
  const auto dimsDataPtr = dims->host<int32_t>();
  int mod[MNN_MAX_TENSOR_DIM];
  OpCommonUtils::computeStride(mod, dimsDataPtr, dimsSize);
  auto outputDataPtr = outputs[0]->host<int32_t>();

  int coordinate[MNN_MAX_TENSOR_DIM];
  for (int i = 0; i < elmentSize; ++i) {
    OpCommonUtils::unravelIndexHelper(coordinate, mod, dimsSize, indicesPtr[i]);
    // assign value
    for (int k = 0; k < dimsSize; ++k) {
      outputDataPtr[i + k * elmentSize] = coordinate[k];
    }
  }
  return Status::OK();
}

class CPUUnravelIndexCreator : public CPUDevice::Creator {
 public:
  virtual Operation* onCreate(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs,
                              const tars::Op* op,
                              Device* backend) const override {
    return new CPUUnravelIndex(backend);
  }
};

REGISTER_CPU_OP_CREATOR(CPUUnravelIndexCreator, OpType_UnravelIndex);

}  // namespace tars

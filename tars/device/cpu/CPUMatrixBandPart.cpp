//
//  CPUMatrixBandPart.cpp
//  MNN
//
//  Created by MNN on 2019/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "device/cpu/CPUMatrixBandPart.hpp"
#include "device/cpu/compute/ConvOpt.h"
namespace tars {
Status CPUMatrixBandPart::onResize(const std::vector<Tensor *> &inputs,
                                   const std::vector<Tensor *> &outputs) {
  MNN_ASSERT(3 == inputs.size());
  auto dimensions = inputs[0]->dimensions();
  auto height = inputs[0]->length(dimensions - 2);
  auto width = inputs[0]->length(dimensions - 1);
  mMask.reset(
      Tensor::createDevice<float>({1, height * width}, Tensor::CAFFE_C4));
  auto res = backend()->onAcquireBuffer(mMask.get(), Device::DYNAMIC);
  if (!res) {
    return Status::ERROR();
  }
  backend()->onReleaseBuffer(mMask.get(), Device::DYNAMIC);
  return Status::OK();
}
Status CPUMatrixBandPart::onExecute(const std::vector<Tensor *> &inputs,
                                    const std::vector<Tensor *> &outputs) {
  // Generate Mask
  auto lower = inputs[1]->host<int32_t>()[0];
  auto upper = inputs[2]->host<int32_t>()[0];
  auto maskPtr = mMask->host<float>();
  auto dimensions = inputs[0]->dimensions();
  auto height = inputs[0]->length(dimensions - 2);
  auto width = inputs[0]->length(dimensions - 1);

  for (int y = 0; y < height; ++y) {
    auto maskY = maskPtr + y * width;
    for (int x = 0; x < width; ++x) {
      bool valid =
          (lower < 0 || (y - x) <= lower) && (upper < 0 || (x - y) <= upper);
      maskY[x] = valid ? 1.0f : 0.0f;
    }
  }

  // Run Mul
  auto outputPtr = outputs[0]->host<float>();
  auto inputPtr = inputs[0]->host<float>();
  int outside = 1;
  for (int i = 0; i < inputs[0]->dimensions() - 2; ++i) {
    outside *= inputs[0]->length(i);
  }
  auto inside = height * width;
  for (int i = 0; i < outside; ++i) {
    MNNMatrixProdCommon(outputPtr + i * inside, inputPtr + i * inside, maskPtr,
                        inside, 0, 0, 0, 1);
  }
  return Status::OK();
}

class CPUMatrixBandPartCreator : public CPUDevice::Creator {
 public:
  virtual Operation *onCreate(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs,
                              const tars::Op *op,
                              Device *backend) const override {
    return new CPUMatrixBandPart(backend);
  }
};

REGISTER_CPU_OP_CREATOR(CPUMatrixBandPartCreator, OpType_MatrixBandPart);
}  // namespace tars

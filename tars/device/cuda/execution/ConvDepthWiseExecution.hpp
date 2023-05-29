//
//  ConvDepthWiseoperation.h
//  MNN
//
//  Created by MNN on 2020/08/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvDepthWiseExecution_hpp
#define ConvDepthWiseExecution_hpp

#include <vector>

#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"
namespace tars {
namespace cuda {

struct constBuffer {
  int pad[2];
  int kernelSize[2];
  int stride[2];
  int dilate[2];
  int inputSize[2];
  int outputSize[2];
  int channel;
  int total;
  int batch;
  float minValue = -65504.0f;
  float maxValue = 65504.0f;
} uConstant;

class ConvDepthWiseExecution : public Operation {
 public:
  struct Resource {
    std::shared_ptr<Tensor> weightTensor;
    std::shared_ptr<Tensor> biasTensor;
    void *mFilter;
    void *mBias;
  };
  ConvDepthWiseExecution(const Op *op, Device *bn,
                         std::shared_ptr<Resource> resource);
  virtual ~ConvDepthWiseExecution();
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 protected:
  std::pair<void *, int> mConstBuffer;
  const Op *mOp;
  int mTotalCount;
  constBuffer parameters;
  std::shared_ptr<Resource> mResource;
};

class DeconvDepthWiseExecution : public ConvDepthWiseExecution {
 public:
  DeconvDepthWiseExecution(const Op *op, Device *bn,
                           std::shared_ptr<Resource> resource)
      : ConvDepthWiseExecution(op, bn, resource) {
    // Do nothing
  }
  virtual ~DeconvDepthWiseExecution() {
    // Do nothing
  }
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
};
}  // namespace cuda
}  // namespace tars
#endif
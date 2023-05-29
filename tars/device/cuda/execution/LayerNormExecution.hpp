//
//  LayerNormoperation.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef LayerNormExecution_hpp
#define LayerNormExecution_hpp

#include <vector>

#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {

class LayerNormExecution : public Operation {
 public:
  LayerNormExecution(const LayerNorm *layer_norm_param, Device *backend);
  virtual ~LayerNormExecution();

  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  CUDARuntime *mRuntime;
  void *mDeviceGamma = nullptr;
  void *mDeviceBeta = nullptr;

  std::vector<int> mAxises;
  int mInside = 1;
  int mOutside = 1;

  float mEps = 0.001;
  int mGroup = 1;

  std::unique_ptr<Tensor> mGammaTensor;
  std::unique_ptr<Tensor> mBetaTensor;

  std::shared_ptr<Tensor> LayerNormTensor;
  std::shared_ptr<Tensor> biasTensor;
};

}  // namespace cuda
}  // namespace tars
#endif /* LayerNormExecution_hpp */

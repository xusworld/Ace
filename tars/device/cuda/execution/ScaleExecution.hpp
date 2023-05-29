//
//  Scaleoperation.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ScaleExecution_hpp
#define ScaleExecution_hpp

#include <vector>

#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {

class ScaleExecution : public Operation {
 public:
  ScaleExecution(const Scale *scale, Device *backend);
  virtual ~ScaleExecution();

  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  CUDARuntime *mRuntime;
  void *mDeviceBias = nullptr;
  void *mDeviceScale = nullptr;
  int mCount;
  int mChannel;
  int mArea;
  std::pair<void *, int> mScaleBiasStorage;
};

}  // namespace cuda
}  // namespace tars
#endif /* ScaleExecution_hpp */

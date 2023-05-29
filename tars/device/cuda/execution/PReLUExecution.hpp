//
//  PReLUoperation.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef PReLUExecution_hpp
#define PReLUExecution_hpp

#include <vector>

#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {

class PReLUExecution : public Operation {
 public:
  PReLUExecution(const PRelu *prelu, Device *backend);
  virtual ~PReLUExecution();

  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  CUDARuntime *mRuntime;
  void *mDeviceSlope = nullptr;
  int mCount;
  int mChannel;
  int mArea;
  std::pair<void *, int> mPreluStorage;
  bool mIsChannelShared = false;
};

}  // namespace cuda
}  // namespace tars
#endif /* PReLUExecution_hpp */

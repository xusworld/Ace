//
//  Pooloperation.h
//  MNN
//
//  Created by MNN on 2020/07/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef PoolExecution_hpp
#define PoolExecution_hpp
#include <vector>

#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {
class PoolExecution : public Operation {
 public:
  PoolExecution(const Pool *pool, Device *backend) : Operation(backend) {
    mParameter = pool;
  }
  virtual ~PoolExecution() {
    // Do nothing
  }
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  const Pool *mParameter;
  std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
  std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
  PoolType mPoolType;
  PoolPadType mPadType;
  std::vector<int> mStrides{1, 1};
  std::vector<int> mKernels{1, 1};
  std::vector<int> mPaddings{0, 0};
};

};  // namespace cuda
};  // namespace tars

#endif
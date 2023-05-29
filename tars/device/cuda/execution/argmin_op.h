//
//  ArgMinoperation.h
//  MNN
//
//  Created by MNN on 2022/06/29.
//  Copyright © 2018 - 2022, Alibaba Group Holding Limited
//

#ifndef ArgMinExecution_hpp
#define ArgMinExecution_hpp
#include <vector>

#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"
namespace tars {
namespace cuda {

class ArgMinOp : public Operation {
 public:
  ArgMinOp(const Op *op, Device *backend);
  virtual ~ArgMinOp();
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  const Op *mOp;
  int mAxis;
  int mInside;
  int mOutside;
  int mDim;
};
}  // namespace cuda
}  // namespace tars

#endif

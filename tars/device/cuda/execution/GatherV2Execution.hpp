//
//  GatherV2operation.h
//  MNN
//
//  Created by MNN on 2020/07/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GatherV2Execution_hpp
#define GatherV2Execution_hpp
#include <vector>

#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"
namespace tars {
namespace cuda {
class GatherV2Execution : public Operation {
 public:
  GatherV2Execution(const Op *op, Device *backend);
  virtual ~GatherV2Execution();
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  const Op *mOp;
  int mAxis;
  int mInside;
  int mOutside;
  int mInpNum;
  int mOutNum;
};
}  // namespace cuda
}  // namespace tars

#endif

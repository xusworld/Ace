//
//  Unaryoperation.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef UnaryExecution_hpp
#define UnaryExecution_hpp

#include <vector>

#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {

class UnaryExecution : public Operation {
 public:
  UnaryExecution(UnaryOpOperation opType, Device *backend);
  virtual ~UnaryExecution() = default;

  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  CUDARuntime *mRuntime;
  UnaryOpOperation mOpType;
  int mCount;
};

}  // namespace cuda
}  // namespace tars
#endif /* UnaryExecution_hpp */

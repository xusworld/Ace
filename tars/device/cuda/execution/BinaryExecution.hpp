//
//  Binaryoperation.h
//  MNN
//
//  Created by MNN on 2020/07/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef BinaryExecution_hpp
#define BinaryExecution_hpp
#include <vector>

#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"
namespace tars {
namespace cuda {
class BinaryExecution : public Operation {
 public:
  BinaryExecution(int opType, Device *backend, int activationType = 0);
  virtual ~BinaryExecution();
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  int mType;
  int mActivationType;
};
}  // namespace cuda
}  // namespace tars

#endif

//
//  Reductionoperation.h
//  MNN
//
//  Created by MNN on 2020/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ReductionExecution_hpp
#define ReductionExecution_hpp
#include <vector>

#include "ReductionTemplate.cuh"
#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"
namespace tars {
namespace cuda {
class ReductionExecution : public Operation {
 public:
  ReductionExecution(ReductionType opType, int axis, Device *backend);
  virtual ~ReductionExecution();
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;

 private:
  ReductionType mType;
  int mAxis;
  ReduceParam mCpuParam;
  std::pair<void *, int> mParam;
};
}  // namespace cuda
}  // namespace tars

#endif
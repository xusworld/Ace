//
//  Softmaxoperation.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SoftmaxExecution_hpp
#define SoftmaxExecution_hpp

#include <float.h>

#include <vector>

#include "ReductionTemplate.cuh"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {

class SoftmaxExecution : public Operation {
 public:
  SoftmaxExecution(int axis, Device *backend);
  virtual ~SoftmaxExecution();

  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  int mAxis;
  Tensor mStorage;
  bool mNeedUnpackC4;
  ReduceParam mCpuParam;
  std::pair<void *, int> mParam;
};

}  // namespace cuda
}  // namespace tars
#endif /* SoftmaxExecution_hpp */
//
//  ReductionExecution.hpp
//  MNN
//
//  Created by MNN on 2020/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ReductionExecution_hpp
#define ReductionExecution_hpp
#include <vector>

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
namespace ace {
namespace CUDA {
class ReductionExecution : public Execution {
 public:
  ReductionExecution(ReductionType opType, int axis, Backend *backend);
  virtual ~ReductionExecution();
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

 private:
  ReductionType mType;
  int mAxis;
};
}  // namespace CUDA
}  // namespace ace

#endif
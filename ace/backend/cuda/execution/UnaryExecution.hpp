//
//  UnaryExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef UnaryExecution_hpp
#define UnaryExecution_hpp

#include <vector>

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"

namespace ace {
namespace CUDA {

class UnaryExecution : public Execution {
 public:
  UnaryExecution(UnaryOpOperation opType, Backend *backend);
  virtual ~UnaryExecution() = default;

  virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

 private:
  CUDARuntime *mRuntime;
  UnaryOpOperation mOpType;
  int mCount;
};

}  // namespace CUDA
}  // namespace ace
#endif /* UnaryExecution_hpp */

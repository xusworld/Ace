//
//  PReLUExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef PReLUExecution_hpp
#define PReLUExecution_hpp

#include <vector>

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"

namespace ace {
namespace CUDA {

class PReLUExecution : public Execution {
 public:
  PReLUExecution(const PRelu *prelu, Backend *backend);
  virtual ~PReLUExecution();

  virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

 private:
  CUDARuntime *mRuntime;
  void *mDeviceSlope = nullptr;
  int mCount;
  int mBatch;
  int mChannel;
  int mArea;

  std::shared_ptr<Tensor> preluTensor;
  bool mIsChannelShared = false;
};

}  // namespace CUDA
}  // namespace ace
#endif /* PReLUExecution_hpp */

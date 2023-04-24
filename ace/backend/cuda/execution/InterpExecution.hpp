//
//  InterpExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef InterpExecution_hpp
#define InterpExecution_hpp

#include <vector>

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"

namespace ace {
namespace CUDA {

class InterpExecution : public Execution {
 public:
  InterpExecution(const Interp *interp, Backend *backend);
  virtual ~InterpExecution();

  virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

 private:
  CUDARuntime *mRuntime;
  float mWidthOffset;
  float mHeightOffset;
  int mResizeType;
  int mCount;
  int mBatch;
  int mChannel;
  int mInputHeight;
  int mInputWidth;
  int mOutputHeight;
  int mOutputWidth;
  float mScaleHeight;
  float mScaleWidth;
};

}  // namespace CUDA
}  // namespace ace
#endif /* InterpExecution_hpp */

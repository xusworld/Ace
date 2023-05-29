//
//  Interpoperation.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef InterpExecution_hpp
#define InterpExecution_hpp

#include <vector>

#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {

class InterpExecution : public Operation {
 public:
  InterpExecution(const Interp *interp, Device *backend);
  virtual ~InterpExecution();

  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
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

}  // namespace cuda
}  // namespace tars
#endif /* InterpExecution_hpp */

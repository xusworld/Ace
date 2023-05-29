//
//  Int8ToFloatoperation.h
//  MNN
//
//  Created by MNN on 2023/01/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Int8ToFloatExecution_hpp
#define Int8ToFloatExecution_hpp

#include <vector>

#include "core/TensorUtils.hpp"
#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {

class Int8ToFloatExecution : public Operation {
 public:
  Int8ToFloatExecution(Device *backend, const std::vector<Tensor *> &inputs,
                       const tars::Op *param);
  virtual ~Int8ToFloatExecution();

  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  void *mScales;
  int8_t mZeroPoint;
  int mClipBits;
  bool mSingle = false;
  int mChannel;
  int mCount;
  int mArea;
  std::pair<void *, int> mScaleStorage;
};

}  // namespace cuda
}  // namespace tars
#endif /* Int8ToFloatExecution_hpp */
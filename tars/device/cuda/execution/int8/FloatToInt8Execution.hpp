//
//  FloatToInt8operation.h
//  MNN
//
//  Created by MNN on 2023/01/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef FloatToInt8Execution_hpp
#define FloatToInt8Execution_hpp

#include <vector>

#include "core/TensorUtils.hpp"
#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {

class FloatToInt8Execution : public Operation {
 public:
  FloatToInt8Execution(Device *backend, const std::vector<Tensor *> &inputs,
                       const tars::Op *param);
  virtual ~FloatToInt8Execution();

  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  void *mScales;
  int8_t mZeroPoint;
  int8_t mClampMin;
  int8_t mClampMax;
  int mClipBits;
  bool mSingle = false;
  int mChannel;
  int mCount;
  int mArea;
  std::pair<void *, int> mScaleStorage;
};

}  // namespace cuda
}  // namespace tars
#endif /* FloatToInt8Execution_hpp */
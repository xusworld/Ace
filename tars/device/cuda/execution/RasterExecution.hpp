//
//  Rasteroperation.h
//  MNN
//
//  Created by MNN on b'2020/04/02'.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef RasterExecution_hpp
#define RasterExecution_hpp
#include <map>
#include <set>

#include "core/TensorUtils.hpp"
#include "device/cuda/core/CUDABackend.hpp"
namespace tars {
namespace cuda {
class RasterExecution : public Operation {
 public:
  RasterExecution(Device *bn) : Operation(bn) {
    // Do nothing
  }
  virtual ~RasterExecution() {
    // Do nothing
  }

  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  void executeFaster(const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs) const;

 private:
  std::map<Tensor *, std::shared_ptr<Tensor>> mTempInput;
  std::vector<std::pair<const Tensor *, Tensor::InsideDescribe::Region *>>
      mTempInputCopy;
  std::vector<std::pair<const Tensor *, Tensor::InsideDescribe::Region>>
      mFastBlit;
  std::shared_ptr<Tensor> mTempOutput;
  Tensor *mOutputPtr;
  bool mNeedZero = false;
  bool mFast = false;
  int mSingleConvert = 0;
  int32_t mZeroPoint = 0;
  // First: type, 0: not , 1: unit, 4:unitc4
  // Second: count
  std::pair<int, int> mFuseRaster;
  void *mOffset;
  std::shared_ptr<Tensor> offsetTensor;
};
}  // namespace cuda
}  // namespace tars
#endif

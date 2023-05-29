//
//  CPUQuantizedAvgPool.hpp
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUQuantizedAvgPool_HPP
#define CPUQuantizedAvgPool_HPP

#include "MNN_generated.h"
#include "core/operation.h"

namespace tars {

class CPUQuantizedAvgPool : public Operation {
 public:
  CPUQuantizedAvgPool(Device *backend, const Op *CPUQuantizedAvgPoolOp);
  virtual ~CPUQuantizedAvgPool() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;

 private:
  int32_t mKernelWidth;
  int32_t mKernelHeight;
  int32_t mPadWidth;
  int32_t mPadHeight;
  int32_t mStrideWidth;
  int32_t mStrideHeight;
  PoolPadType mPadMode;
  int mOutputActivationMin;
  int mOutputActivationMax;
  bool mIstflite;
  std::vector<int> mInputDims;
  std::vector<int> mOutputDims;
};
}  // namespace tars

#endif /* CPUQuantizedAvgPool.hpp */

//
//  CPUQuantizedMaxPool.hpp
//  MNN
//
//  Created by MNN on 2018/08/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUQUANTIZEDMAXPOOL_HPP
#define CPUQUANTIZEDMAXPOOL_HPP

#include "MNN_generated.h"
#include "core/operation.h"

namespace tars {

class CPUQuantizedMaxPool : public Operation {
 public:
  CPUQuantizedMaxPool(Device *backend, const Op *quantizedMaxPoolOp);
  virtual ~CPUQuantizedMaxPool() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  int32_t mKernelWidth;
  int32_t mKernelHeight;
  int32_t mPadWidth;
  int32_t mPadHeight;
  int32_t mStrideWidth;
  int32_t mStrideHeight;
  PoolPadType mPadMode;
};
}  // namespace tars

#endif /* CPUQuantizedMaxPool.hpp */

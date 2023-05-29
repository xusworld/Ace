//
//  CPUTopKV2.hpp
//  MNN
//
//  Created by MNN on 2018/08/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUTOPKV2_HPP
#define CPUTOPKV2_HPP

#include "MNN_generated.h"
#include "core/operation.h"

namespace tars {
class CPUTopKV2 : public Operation {
 public:
  CPUTopKV2(Device *b, const Op *op);
  virtual ~CPUTopKV2() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  bool mLargest = true;
};
}  // namespace tars

#endif  // CPUTOPKV2_HPP

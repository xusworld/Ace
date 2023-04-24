//
//  CPUTopKV2.hpp
//  MNN
//
//  Created by MNN on 2018/08/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUTOPKV2_HPP
#define CPUTOPKV2_HPP

#include "ace_generated.h"
#include "core/Execution.hpp"

namespace ace {
class CPUTopKV2 : public Execution {
 public:
  CPUTopKV2(Backend *b);
  virtual ~CPUTopKV2() = default;
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

 private:
};
}  // namespace ace

#endif  // CPUTOPKV2_HPP

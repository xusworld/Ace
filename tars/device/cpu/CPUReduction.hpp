//
//  CPUReduction.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUReduction_hpp
#define CPUReduction_hpp

#include "device/cpu/CPUDevice.h"

namespace tars {
class CPUReductionCreator : public CPUDevice::Creator {
 public:
  static Operation* create(const std::vector<Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs,
                           const tars::Op* op, Device* backend);
  virtual Operation* onCreate(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs,
                              const tars::Op* op,
                              Device* backend) const override;
};
}  // namespace tars
#endif /* CPUReduction_hpp */

//
//  CPUReduction.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUReduction_hpp
#define CPUReduction_hpp

#include "backend/cpu/CPUBackend.hpp"

namespace ace {
class CPUReductionCreator : public CPUBackend::Creator {
 public:
  static Execution* create(const std::vector<Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs,
                           const ace::Op* op, Backend* backend);
  virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs,
                              const ace::Op* op,
                              Backend* backend) const override;
};
}  // namespace ace
#endif /* CPUReduction_hpp */

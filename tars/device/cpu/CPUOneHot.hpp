//
//  CPUOneHot.hpp
//  MNN
//
//  Created by MNN on 2019/11/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUOneHot_hpp
#define CPUOneHot_hpp

#include "core/operation.h"

namespace tars {

class CPUOneHot : public Operation {
 public:
  CPUOneHot(Device *b, int axis) : Operation(b), mAxis(axis) {}
  virtual ~CPUOneHot() = default;

  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  int mAxis;
};

}  // namespace tars

#endif /* CPUOneHot_hpp */

//
//  CPULinSpace.hpp
//  MNN
//
//  Created by MNN on 2019/12/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPULinSpace_hpp
#define CPULinSpace_hpp

#include "core/operation.h"

namespace tars {
class CPULinSpace : public Operation {
 public:
  CPULinSpace(Device *b) : Operation(b) {
    // nothing to do
  }
  virtual ~CPULinSpace() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
};

}  // namespace tars

#endif /* CPULinSpace_hpp */

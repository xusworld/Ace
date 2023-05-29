//
//  CPUWhere.hpp
//  MNN
//
//  Created by MNN on 2018/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUWhere_hpp
#define CPUWhere_hpp

#include "core/operation.h"

namespace tars {
class CPUWhere : public Operation {
 public:
  CPUWhere(Device *b) : Operation(b) {
    // nothing to do
  }
  virtual ~CPUWhere() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
};
}  // namespace tars

#endif /* CPUWhere_hpp */

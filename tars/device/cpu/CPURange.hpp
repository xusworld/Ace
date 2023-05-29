//
//  CPURange.hpp
//  MNN
//
//  Created by MNN on 2018/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPURange_hpp
#define CPURange_hpp

#include "core/operation.h"

namespace tars {
template <typename T>
class CPURange : public Operation {
 public:
  CPURange(Device *backend);
  virtual ~CPURange() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
};

}  // namespace tars

#endif /* CPURange.hpp */

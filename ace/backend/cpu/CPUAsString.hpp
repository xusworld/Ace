//
//  CPUAsString.hpp
//  MNN
//
//  Created by MNN on 2018/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUAsString_hpp
#define CPUAsString_hpp

#include "backend/cpu/CPUBackend.hpp"

namespace ace {
class CPUAsStringCreator : public CPUBackend::Creator {
 public:
  virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs,
                              const ace::Op* op,
                              Backend* backend) const override;
};
}  // namespace ace
#endif /* CPUAsString_hpp */

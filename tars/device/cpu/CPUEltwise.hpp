//
//  CPUEltwise.hpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUEltwise_hpp
#define CPUEltwise_hpp

#include "MNN_generated.h"
#include "core/operation.h"

namespace tars {
class CPUEltwise : public Operation {
 public:
  CPUEltwise(Device *b, EltwiseType type, std::vector<float> coef);
  virtual ~CPUEltwise() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  EltwiseType mType;
  std::vector<float> mCoeff;
};

}  // namespace tars

#endif /* CPUEltwise_hpp */

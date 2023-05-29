//
//  CPUInstanceNorm.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUInstanceNorm_hpp
#define CPUInstanceNorm_hpp

#include "core/AutoStorage.h"
#include "core/operation.h"

namespace tars {
class CPUInstanceNorm : public Operation {
 public:
  CPUInstanceNorm(Device *backend, const tars::Op *op);
  virtual ~CPUInstanceNorm() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  AutoStorage<float> mScale;
  AutoStorage<float> mBias;
  float mEpsilon;
};
}  // namespace tars

#endif /* CPUInstanceNorm_hpp */

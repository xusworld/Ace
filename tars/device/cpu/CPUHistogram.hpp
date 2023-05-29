//
//  CPUHistogram.hpp
//  MNN
//
//  Created by MNN on 2022/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUHistogram_hpp
#define CPUHistogram_hpp

#include "core/operation.h"

namespace tars {
class CPUHistogram : public Operation {
 public:
  CPUHistogram(Device *backend, const Op *op);
  virtual ~CPUHistogram() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  template <typename T>
  Status histogram(Tensor *input, Tensor *output);
  int mChannel, mBinNum, mMin, mMax, mSize, mStride;
  float mAlpha, mBeta;
};

}  // namespace tars

#endif /* CPUHistogram_hpp */

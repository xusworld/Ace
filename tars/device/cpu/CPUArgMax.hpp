//
//  CPUArgMax.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUArgMax_hpp
#define CPUArgMax_hpp

#include "core/operation.h"

namespace tars {

class CPUArgMax : public Operation {
 public:
  enum ArgMinOrMax { ARGMIN, ARGMAX };
  CPUArgMax(Device *backend, ArgMinOrMax mode, int topk, int outMaxVal,
            int softmaxThreshold, int axis);
  virtual ~CPUArgMax() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  Tensor mInputBuffer;
  Tensor mOutputBuffer;
  int mTopk;
  int mOutMaxVal;
  int mSoftmaxThreshold;
  int mAxis;
  int mNum;
  int mDim;
  int mKeyExtent;
  bool mFromNHWC;
  ArgMinOrMax mMode;
};

}  // namespace tars

#endif /* CPUArgMax_hpp */

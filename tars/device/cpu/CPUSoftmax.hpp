//
//  CPUSoftmax.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSoftmax_hpp
#define CPUSoftmax_hpp

#include "core/operation.h"

namespace tars {
class CPUSoftmax : public Operation {
 public:
  CPUSoftmax(Device *b, int axis);
  virtual ~CPUSoftmax() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  static Operation *create(const tars::Op *op, Device *backend);

 private:
  int _softmaxCommon(const float *srcData, float *dstData, int inside,
                     int outside, int channel, float *maxValue, float *sumValue,
                     int threadNum);
  int _softmax1(const float *srcData, float *dstData, int outside, int channel,
                int threadNum);

  int mAxis;
  Tensor mStorage;
  Tensor mMaxValue;
  Tensor mSumValue;
  bool mNeedUnpackC4;
};
}  // namespace tars

#endif /* CPUSoftmax_hpp */

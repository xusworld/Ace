//
//  CPUMoments.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUMoments_hpp
#define CPUMoments_hpp

#include "core/operation.h"

namespace tars {

class CPUMoments : public Operation {
 public:
  CPUMoments(Device* backend, const tars::Op* op);
  virtual ~CPUMoments() = default;
  virtual Status onExecute(const std::vector<Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs) override;
  virtual Status onResize(const std::vector<Tensor*>& inputs,
                          const std::vector<Tensor*>& outputs) override;

 private:
  void CalculateMean(const float* src, float* dst, int batch, int channelDiv4,
                     int inImageSize, int inBatchStride, int outBatchStride);
  std::vector<int> mAxis;
  bool mKeepDims;
  std::shared_ptr<Tensor> mMidBuffer;
};

}  // namespace tars

#endif /* CPUMoments_hpp */

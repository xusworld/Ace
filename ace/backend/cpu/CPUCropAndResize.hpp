//
//  CPUCropAndResize.hpp
//  MNN
//
//  Created by MNN on 2018/08/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUCropAndResize_hpp
#define CPUCropAndResize_hpp

#include "ace_generated.h"
#include "core/Execution.hpp"

namespace ace {
template <typename T>
class CPUCropAndResize : public Execution {
 public:
  CPUCropAndResize(Backend* backend, const Op* op);
  ~CPUCropAndResize() = default;
  virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs) override;

 private:
  const ErrorCode CropAndResize(const Tensor* image, const Tensor* boxes,
                                const Tensor* boxIndex, Tensor* crops);
  CropAndResizeMethod mMethod;
  float mExtrapolationValue;
};

}  // namespace ace
#endif /* CPUCropAndResize_hpp */

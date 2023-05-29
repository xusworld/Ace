//
//  CPUCropAndResize.hpp
//  MNN
//
//  Created by MNN on 2018/08/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUCropAndResize_hpp
#define CPUCropAndResize_hpp

#include "MNN_generated.h"
#include "core/operation.h"

namespace tars {
template <typename T>
class CPUCropAndResize : public Operation {
 public:
  CPUCropAndResize(Device* backend, const Op* op);
  ~CPUCropAndResize() = default;
  virtual Status onExecute(const std::vector<Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs) override;

 private:
  const Status CropAndResize(const Tensor* image, const Tensor* boxes,
                             const Tensor* boxIndex, Tensor* crops);
  CropAndResizeMethod mMethod;
  float mExtrapolationValue;
};

}  // namespace tars
#endif /* CPUCropAndResize_hpp */

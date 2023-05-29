//
//  CPUCast.hpp
//  MNN
//
//  Created by MNN on 2018/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUCast_hpp
#define CPUCast_hpp

#include "device/cpu/CPUDevice.h"

namespace tars {
class CPUCastCreator : public CPUDevice::Creator {
 public:
  enum ConvertType {
    INT8_TO_FlOAT = 0,
    FlOAT_TO_INT8 = 1,
  };
  virtual Operation* onCreate(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs,
                              const tars::Op* op,
                              Device* backend) const override;
  static Status cast(const Tensor* input, const Tensor* output,
                     const CPUDevice* bn, ConvertType type);
  static Status cast(void* const inputRaw, void* outputRaw, ConvertType type,
                     int number, float scale, float zero, float min, float max,
                     const CPUDevice* bn);
};
}  // namespace tars
#endif /* CPUCast_hpp */

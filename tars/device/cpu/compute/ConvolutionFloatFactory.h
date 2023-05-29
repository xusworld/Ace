//
//  ConvolutionFloatFactory.h
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionFloatFactory_h
#define ConvolutionFloatFactory_h

#include "device/cpu/CPUDevice.h"

namespace tars {
class ConvolutionFloatFactory {
 public:
  static Operation* create(const std::vector<Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs,
                           const tars::Op* op, Device* backend);
};
}  // namespace tars

#endif /* ConvolutionFloatFactory_hpp */

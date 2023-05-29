//
//  ConvolutionIntFactory.hpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionIntFactory_hpp
#define ConvolutionIntFactory_hpp

#include <stdint.h>

#include <memory>

#include "core/ConvolutionCommon.hpp"
#include "device/cpu/CPUDevice.h"

namespace tars {
class ConvolutionIntFactory {
 public:
  static Operation* create(const Tensor* input, const Tensor* output,
                           const tars::Op* op, Device* backend,
                           const ConvolutionCommon::Int8Common* common);

  static Operation* createUnit(const Tensor* input, const Tensor* output,
                               const tars::Op* op, Device* bn,
                               const ConvolutionCommon::Int8Common* common,
                               const float* bias, size_t biasSize);
};
}  // namespace tars

#endif /* ConvolutionIntFactory_hpp */

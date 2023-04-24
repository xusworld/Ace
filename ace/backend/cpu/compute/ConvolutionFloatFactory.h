//
//  ConvolutionFloatFactory.h
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionFloatFactory_h
#define ConvolutionFloatFactory_h

#include "backend/cpu/CPUBackend.hpp"

namespace ace {
class ConvolutionFloatFactory {
 public:
  static Execution* create(const std::vector<Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs,
                           const ace::Op* op, Backend* backend);
};
}  // namespace ace

#endif /* ConvolutionFloatFactory_hpp */

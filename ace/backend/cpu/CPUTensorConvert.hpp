//
//  CPUTensorConvert.hpp
//  MNN
//
//  Created by MNN on 2018/08/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUTensorConvert_hpp
#define CPUTensorConvert_hpp

#include "Tensor_generated.h"
#include "compute/CommonOptFunction.h"
#include "core/Execution.hpp"
namespace ace {

class CPUTensorConverter {
 public:
  static std::tuple<int, int, int> splitDimensions(const halide_buffer_t& ib,
                                                   DATA_FORMAT source);
  static ErrorCode convert(const Tensor* input, const Tensor* output,
                           const CoreFunctions* core = nullptr);
  static ErrorCode convert(const void* inputRaw, void* outputRaw,
                           DATA_FORMAT inputFormat, DATA_FORMAT outputFormat,
                           int batch, int area, int channel, int bytes,
                           const CoreFunctions* core);
};

}  // namespace ace

#endif /* CPUTensorConvert_hpp */

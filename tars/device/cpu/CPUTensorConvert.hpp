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
#include "core/operation.h"
namespace tars {

class CPUTensorConverter {
 public:
  static std::tuple<int, int, int> splitDimensions(const halide_buffer_t& ib,
                                                   MNN_DATA_FORMAT source);
  static Status convert(const Tensor* input, const Tensor* output,
                        const CoreFunctions* core = nullptr, int tId = 0,
                        int numberThread = 1);
  static Status convert(const void* inputRaw, void* outputRaw,
                        MNN_DATA_FORMAT inputFormat,
                        MNN_DATA_FORMAT outputFormat, int batch, int area,
                        int channel, int bytes, const CoreFunctions* core,
                        int tId = 0, int numberThread = 1);
};

}  // namespace tars

#endif /* CPUTensorConvert_hpp */

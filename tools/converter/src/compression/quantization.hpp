//
//  quantization.hpp
//  MNN
//
//  Created by MNN on 2020/07/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_CONVERTER_COMPRESSION_QUANTIZATION_HPP_
#define MNN_CONVERTER_COMPRESSION_QUANTIZATION_HPP_

#include <string>
#include <unordered_map>
#include <vector>

#include "MNN_compression.pb.h"

namespace compression {

struct Quantization {
  ace::Compression::QuantizeParams::RoundMode round_mode;

  struct TensorParams {
    int32_t nbit;
    std::vector<float> scale;
    float zero_point;
    float clamp_min;
    float clamp_max;
    ace::Compression::LayerQuantizeParams_QuantMethod method;
  };
  std::unordered_map<std::string, std::vector<TensorParams>> tensors;
};

};  // namespace compression

#endif  // MNN_CONVERTER_COMPRESSION_QUANTIZATION_HPP_

#pragma once

#include "onednn.h"

namespace onednn {

// Ref https://oneapi-src.github.io/oneDNN/dev_guide_eltwise.html.
enum class ElementwiseOpType {
  Abs,
  BoundedRelu,
  Clip,
  ClipV2,
  Elu,
  Exp,
  GeluErf,
  GeluTanh,
  HardSigmoid,
  HardSwish,
  Linear,
  Log,
  Logistic,
  LogSigmoid,
  Mish,
  Pow,
  Relu,
  Round,
  SoftRelu,
  SoftReluV2,
  Sqrt,
  Swish,
  Tanh,
};

dnnl::algorithm ElementwiseOpToAlgoKind(const ElementwiseOpType& optype);

enum class CoreDataType {
  FP32,
  DF64,
  FP16,
  BF16,
  INT8,
  S32,
  S8,
  U8,
};

static inline dnnl::memory::data_type GetOneDNNDataType(
    const CoreDataType& dt) {
  switch (dt) {
    case CoreDataType::FP32:
      return dnnl::memory::data_type::f32;
    case CoreDataType::FP16:
      return dnnl::memory::data_type::f16;
    case CoreDataType::BF16:
      return dnnl::memory::data_type::bf16;
    case CoreDataType::S8:
      return dnnl::memory::data_type::s8;
    case CoreDataType::U8:
      return dnnl::memory::data_type::u8;
    default:
      std::cout << "DataType not valid.";
      break;
  }
  return dnnl::memory::data_type::f32;
}

std::pair<dnnl::primitive, std::unordered_map<int, dnnl::memory>>
CreateElementwisePrimitive(const ElementwiseOpType& optype, float* dst,
                           float* src, const std::vector<int64_t> shape,
                           const float alpha, const float beta);

void ElementwisePrimitiveRunner(const dnnl::primitive&,
                                const std::unordered_map<int, dnnl::memory>&);

}  // namespace onednn
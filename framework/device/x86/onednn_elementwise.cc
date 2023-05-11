#include "onednn_elementwise.h"

#include <oneapi/dnnl/dnnl_debug.h>
#include <sys/time.h>

#include <functional>
#include <numeric>
#include <oneapi/dnnl/dnnl.hpp>
#include <utility>

namespace onednn {

dnnl::algorithm ElementwiseOpToAlgoKind(const ElementwiseOpType& optype) {
  switch (optype) {
    case ElementwiseOpType::Abs:
      return dnnl::algorithm::eltwise_abs;
    case ElementwiseOpType::BoundedRelu:
      return dnnl::algorithm::eltwise_bounded_relu;
    case ElementwiseOpType::Clip:
      return dnnl::algorithm::eltwise_clip;
    case ElementwiseOpType::ClipV2:
      return dnnl::algorithm::eltwise_clip_v2;
    case ElementwiseOpType::Elu:
      return dnnl::algorithm::eltwise_elu;
    case ElementwiseOpType::Exp:
      return dnnl::algorithm::eltwise_exp;
    case ElementwiseOpType::GeluErf:
      return dnnl::algorithm::eltwise_gelu_erf;
    case ElementwiseOpType::GeluTanh:
      return dnnl::algorithm::eltwise_gelu_tanh;
    case ElementwiseOpType::HardSigmoid:
      return dnnl::algorithm::eltwise_hardsigmoid;
    case ElementwiseOpType::HardSwish:
      return dnnl::algorithm::eltwise_hardswish;
    case ElementwiseOpType::Linear:
      return dnnl::algorithm::eltwise_linear;
    case ElementwiseOpType::Log:
      return dnnl::algorithm::eltwise_log;
    case ElementwiseOpType::Logistic:
      return dnnl::algorithm::eltwise_logistic;
    case ElementwiseOpType::LogSigmoid:
      return dnnl::algorithm::eltwise_logsigmoid;
    case ElementwiseOpType::Mish:
      return dnnl::algorithm::eltwise_mish;
    case ElementwiseOpType::Pow:
      return dnnl::algorithm::eltwise_pow;
    case ElementwiseOpType::Relu:
      return dnnl::algorithm::eltwise_relu;
    case ElementwiseOpType::Round:
      return dnnl::algorithm::eltwise_round;
    case ElementwiseOpType::SoftRelu:
      return dnnl::algorithm::eltwise_soft_relu;
    case ElementwiseOpType::SoftReluV2:
      return dnnl::algorithm::eltwise_soft_relu_v2;
    case ElementwiseOpType::Sqrt:
      return dnnl::algorithm::eltwise_sqrt;
    case ElementwiseOpType::Swish:
      return dnnl::algorithm::eltwise_swish;
    case ElementwiseOpType::Tanh:
      return dnnl::algorithm::eltwise_tanh;
    default:
      break;
  }
  return dnnl::algorithm::eltwise_abs;
}

std::pair<dnnl::primitive, std::unordered_map<int, dnnl::memory>>
CreateElementwisePrimitive(const ElementwiseOpType& optype, float* dst,
                           float* src, const std::vector<int64_t> shape,
                           const float alpha = 0.0f, const float beta = 0.0f) {
  CHECK(dst != nullptr && src != nullptr);

  // Create execution dnnl::engine.
  dnnl::engine engine = DnnlRuntime::GetDnnlEngine();
  // Create dnnl::stream.
  dnnl::stream stream = DnnlRuntime::GetDnnlStream();

  // oneDNN data type.
  auto data_type = GetOneDNNDataType(CoreDataType::FP32);

  // Create src and dst dims.
  dnnl::memory::dims src_dims = shape;
  dnnl::memory::dims dst_dims = shape;

  // Create src and dst memory descriptors and memory objects.
  auto src_md =
      dnnl::memory::desc(src_dims, data_type, dnnl::memory::format_tag::nchw);

  auto dst_md =
      dnnl::memory::desc(dst_dims, data_type, dnnl::memory::format_tag::nchw);

  auto src_mem = dnnl::memory(src_md, engine, src);
  auto dst_mem = dnnl::memory(dst_md, engine, dst);

  // Create operation descriptor.
  auto eltwise_d = dnnl::eltwise_forward::desc(
      dnnl::prop_kind::forward_inference, ElementwiseOpToAlgoKind(optype),
      src_md, alpha, beta);

  // Create primitive descriptor.
  auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(eltwise_d, engine);

  // Create the primitive
  auto primitive = dnnl::eltwise_forward(eltwise_pd);
  // Primitive arguments.
  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, src_mem});
  args.insert({DNNL_ARG_DST, dst_mem});
  return std::make_pair(primitive, args);
}

void ElementwisePrimitiveRunner(
    const dnnl::primitive& primitive,
    const std::unordered_map<int, dnnl::memory>& args) {
  // try {
  // Create dnnl::stream.
  dnnl::stream stream = DnnlRuntime::GetDnnlStream();
  // Primitive execution: element-wise.
  primitive.execute(stream, args);

  // Wait for the computation to finalize.
  stream.wait();
  // std::cout << "timecost: " << timecost << std::endl;
  // } catch (dnnl::error& e) {
  //   std::cout << "oneDNN error caught: " << std::endl
  //             << "\tMessage: " << e.what() << std::endl;
  // }
}

}  // namespace onednn
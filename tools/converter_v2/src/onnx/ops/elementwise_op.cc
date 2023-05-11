#include <stdio.h>

#include "../op_converter.h"
#include "ace/ir/op_generated.h"
#include "ace/ir/op_option_generated.h"

namespace ace {
namespace model {

namespace {
void OnnxElementwiseNodeToOp(
    ace::OpT *op, const onnx::NodeProto *node,
    std::vector<const onnx::TensorProto *> initializers) {
  const auto &type = node->op_type();
  if (type == "Abs") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "BoundedRelu") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Clip") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "ClipV2") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "ClippedRelu") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Elu") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Exp") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "GeluTanh") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "HardSigmoid") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "HardSwish") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "LeakyRelu") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Linear") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Log") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Logistic") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "LogSigmoid") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Mish") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Pow") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "PRelu") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Relu") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Relu6") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Round") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Selu") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Sigmoid") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "SoftRelu") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "SoftReluV2") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Sqrt") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Swish") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else if (type == "Tanh") {
    auto option = new ace::ReluOptionT;
    op->option.value = option;
    op->type = ace::OpType_Relu;
  } else {
    LOG(FATAL) << "Elementwise type: " << type << " not support yet.";
  }
}
}  // namespace

#define REGISTER_ELEMENTWISE_OP_CONVERTER(name)                           \
  DECLARE_ONNX_NODE_PARSER(name##OnnxNodeParser);                         \
  ace::OpType name##OnnxNodeParser::type() { return ace::OpType_##name; } \
  void name##OnnxNodeParser::parse(                                       \
      ace::OpT *op, const onnx::NodeProto *node,                          \
      std::vector<const onnx::TensorProto *> initializers) {              \
    OnnxElementwiseNodeToOp(op, node, initializers);                      \
  }                                                                       \
  static OnnxNodeParserRegister<name##OnnxNodeParser> _##name##op_parser(#name);

REGISTER_ELEMENTWISE_OP_CONVERTER(Abs)
REGISTER_ELEMENTWISE_OP_CONVERTER(BoundedRelu)
REGISTER_ELEMENTWISE_OP_CONVERTER(Clip)
REGISTER_ELEMENTWISE_OP_CONVERTER(ClipV2)
REGISTER_ELEMENTWISE_OP_CONVERTER(ClippedRelu)
REGISTER_ELEMENTWISE_OP_CONVERTER(Elu)
REGISTER_ELEMENTWISE_OP_CONVERTER(Exp)
REGISTER_ELEMENTWISE_OP_CONVERTER(GeluTanh)
REGISTER_ELEMENTWISE_OP_CONVERTER(HardSigmoid)
REGISTER_ELEMENTWISE_OP_CONVERTER(HardSwish)
REGISTER_ELEMENTWISE_OP_CONVERTER(LeakyRelu)
REGISTER_ELEMENTWISE_OP_CONVERTER(Linear)
REGISTER_ELEMENTWISE_OP_CONVERTER(Log)
REGISTER_ELEMENTWISE_OP_CONVERTER(Logistic)
REGISTER_ELEMENTWISE_OP_CONVERTER(LogSigmoid)
REGISTER_ELEMENTWISE_OP_CONVERTER(Mish)
REGISTER_ELEMENTWISE_OP_CONVERTER(Pow)
REGISTER_ELEMENTWISE_OP_CONVERTER(PRelu)
REGISTER_ELEMENTWISE_OP_CONVERTER(Relu)
REGISTER_ELEMENTWISE_OP_CONVERTER(Relu6)
REGISTER_ELEMENTWISE_OP_CONVERTER(Round)
REGISTER_ELEMENTWISE_OP_CONVERTER(Selu)
REGISTER_ELEMENTWISE_OP_CONVERTER(Sigmoid)
REGISTER_ELEMENTWISE_OP_CONVERTER(SoftRelu)
REGISTER_ELEMENTWISE_OP_CONVERTER(SoftReluV2)
REGISTER_ELEMENTWISE_OP_CONVERTER(Sqrt)
REGISTER_ELEMENTWISE_OP_CONVERTER(Swish)
REGISTER_ELEMENTWISE_OP_CONVERTER(Tanh)

}  // namespace model
}  // namespace ace
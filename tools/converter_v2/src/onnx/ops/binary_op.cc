#include <stdio.h>

#include "../op_converter.h"
#include "ace/ir/op_generated.h"
#include "ace/ir/op_option_generated.h"

namespace ace {
namespace model {

namespace {

void OnnxBinaryNodeToOp(ace::OpT* op, const onnx::NodeProto* node,
                        std::vector<const onnx::TensorProto*> initializers) {
  const auto& type = node->op_type();
  if (type == "Add") {
    auto option = new ace::AddOptionT;
    op->option.value = option;
    op->type = ace::OpType_Add;
  } else if (type == "Mul") {
    auto option = new ace::MulOptionT;
    op->option.value = option;
    op->type = ace::OpType_Mul;
  } else if (type == "Min") {
    auto option = new ace::MinOptionT;
    op->option.value = option;
    op->type = ace::OpType_Min;
  } else if (type == "Max") {
    auto option = new ace::MaxOptionT;
    op->option.value = option;
    op->type = ace::OpType_Max;
  } else if (type == "Div") {
    auto option = new ace::DivOptionT;
    op->option.value = option;
    op->type = ace::OpType_Div;
  } else if (type == "Greater") {
    auto option = new ace::GreaterThanOptionT;
    op->option.value = option;
    op->type = ace::OpType_GreaterThan;
  } else if (type == "GreaterOrEqual") {
    auto option = new ace::GreaterEqualOptionT;
    op->option.value = option;
    op->type = ace::OpType_GreaterEqual;
  } else if (type == "Less") {
    auto option = new ace::LessThanOptionT;
    op->option.value = option;
    op->type = ace::OpType_LessThan;
  } else if (type == "LessOrEqual") {
    auto option = new ace::LessEqualOptionT;
    op->option.value = option;
    op->type = ace::OpType_LessEqual;
  } else if (type == "Equal") {
    auto option = new ace::EqualToOptionT;
    op->option.value = option;
    op->type = ace::OpType_EqualTo;
  } else if (type == "Less") {
    auto option = new ace::LessThanOptionT;
    op->option.value = option;
    op->type = ace::OpType_LessThan;
  } else {
    LOG(FATAL) << "Binary type: " << type << " not support yet.";
  }
}
}  // namespace

#define REGISTER_BINARY_OP_CONVERTER(name)                                \
  DECLARE_ONNX_NODE_PARSER(name##OnnxNodeParser);                         \
  ace::OpType name##OnnxNodeParser::type() { return ace::OpType_##name; } \
  void name##OnnxNodeParser::parse(                                       \
      ace::OpT* op, const onnx::NodeProto* node,                          \
      std::vector<const onnx::TensorProto*> initializers) {               \
    OnnxBinaryNodeToOp(op, node, initializers);                           \
  }                                                                       \
  static OnnxNodeParserRegister<name##OnnxNodeParser> _##name##op_parser(#name);

REGISTER_BINARY_OP_CONVERTER(Add)
REGISTER_BINARY_OP_CONVERTER(Mul)
REGISTER_BINARY_OP_CONVERTER(Min)
REGISTER_BINARY_OP_CONVERTER(Max)
REGISTER_BINARY_OP_CONVERTER(Div)
REGISTER_BINARY_OP_CONVERTER(GreaterThan)
REGISTER_BINARY_OP_CONVERTER(GreaterEqual)
REGISTER_BINARY_OP_CONVERTER(LessThan)
REGISTER_BINARY_OP_CONVERTER(LessEqual)
REGISTER_BINARY_OP_CONVERTER(EqualTo)

}  // namespace model
}  // namespace ace

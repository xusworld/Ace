#include <stdio.h>

#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {
DECLARE_ONNX_NODE_PARSER(BinaryOpOnnx);

ace::OpType BinaryOpOnnx::opType() { return ace::OpType_BinaryOp; }

ace::OpParameter BinaryOpOnnx::type() { return ace::OpParameter_BinaryOp; }

void BinaryOpOnnx::parse(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                         std::vector<const onnx::TensorProto*> initializers) {
  const auto& originalType = onnxNode->op_type();
  auto param = new ace::BinaryOpT;
#define TO_BINARY_OP(src, dst) \
  if (originalType == src) {   \
    param->opType = dst;       \
  }

  TO_BINARY_OP("Add", ace::BinaryOpOperation_ADD);
  TO_BINARY_OP("And", ace::BinaryOpOperation_MUL);
  TO_BINARY_OP("Div", ace::BinaryOpOperation_REALDIV);
  TO_BINARY_OP("Mul", ace::BinaryOpOperation_MUL);
  TO_BINARY_OP("Equal", ace::BinaryOpOperation_EQUAL);
  TO_BINARY_OP("Less", ace::BinaryOpOperation_LESS);
  TO_BINARY_OP("LessOrEqual", ace::BinaryOpOperation_LESS_EQUAL);
  TO_BINARY_OP("Greater", ace::BinaryOpOperation_GREATER);
  TO_BINARY_OP("GreaterOrEqual", ace::BinaryOpOperation_GREATER_EQUAL);
  TO_BINARY_OP("Max", ace::BinaryOpOperation_MAXIMUM);
  TO_BINARY_OP("Min", ace::BinaryOpOperation_MINIMUM);
  // TODO: Consified fmod case
  TO_BINARY_OP("Mod", ace::BinaryOpOperation_MOD);
  TO_BINARY_OP("Pow", ace::BinaryOpOperation_POW);
  TO_BINARY_OP("Sub", ace::BinaryOpOperation_SUB);
  TO_BINARY_OP("Sum", ace::BinaryOpOperation_ADD);
  auto type = onnxNode->op_type();
  dstOp->main.value = param;
}

REGISTER_ONNX_NODE_PARSER(BinaryOpOnnx, Add);
REGISTER_ONNX_NODE_PARSER(BinaryOpOnnx, And);
REGISTER_ONNX_NODE_PARSER(BinaryOpOnnx, Sum);
REGISTER_ONNX_NODE_PARSER(BinaryOpOnnx, Sub);
REGISTER_ONNX_NODE_PARSER(BinaryOpOnnx, Div);
REGISTER_ONNX_NODE_PARSER(BinaryOpOnnx, Mul);
REGISTER_ONNX_NODE_PARSER(BinaryOpOnnx, Pow);
REGISTER_ONNX_NODE_PARSER(BinaryOpOnnx, Equal);
REGISTER_ONNX_NODE_PARSER(BinaryOpOnnx, Less);
REGISTER_ONNX_NODE_PARSER(BinaryOpOnnx, LessOrEqual);
REGISTER_ONNX_NODE_PARSER(BinaryOpOnnx, Greater);
REGISTER_ONNX_NODE_PARSER(BinaryOpOnnx, GreaterOrEqual);
REGISTER_ONNX_NODE_PARSER(BinaryOpOnnx, Max);
REGISTER_ONNX_NODE_PARSER(BinaryOpOnnx, Min);
REGISTER_ONNX_NODE_PARSER(BinaryOpOnnx, Mod);

}  // namespace parser
}  // namespace ace

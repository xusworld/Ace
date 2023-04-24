#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(UnaryOnnx);

ace::OpType UnaryOnnx::opType() { return ace::OpType_UnaryOp; }

ace::OpParameter UnaryOnnx::type() { return ace::OpParameter_UnaryOp; }

void UnaryOnnx::parse(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                      std::vector<const onnx::TensorProto *> initializers) {
  std::unique_ptr<ace::UnaryOpT> unaryOpParam(new ace::UnaryOpT);
  unaryOpParam->T = ace::DataType_DT_FLOAT;

  const auto &originalType = onnxNode->op_type();

#define TO_UNARY_OP(src, dst)   \
  if (originalType == src) {    \
    unaryOpParam->opType = dst; \
  }

  TO_UNARY_OP("Abs", ace::UnaryOpOperation_ABS);
  TO_UNARY_OP("Acos", ace::UnaryOpOperation_ACOS);
  TO_UNARY_OP("Acosh", ace::UnaryOpOperation_ACOSH);
  TO_UNARY_OP("Asinh", ace::UnaryOpOperation_ASINH);
  TO_UNARY_OP("Atan", ace::UnaryOpOperation_ATAN);
  TO_UNARY_OP("Atanh", ace::UnaryOpOperation_ATANH);
  TO_UNARY_OP("Asin", ace::UnaryOpOperation_ASIN);
  TO_UNARY_OP("Ceil", ace::UnaryOpOperation_CEIL);
  TO_UNARY_OP("Cos", ace::UnaryOpOperation_COS);
  TO_UNARY_OP("Cosh", ace::UnaryOpOperation_COSH);
  TO_UNARY_OP("Exp", ace::UnaryOpOperation_EXP);
  TO_UNARY_OP("Erf", ace::UnaryOpOperation_ERF);
  TO_UNARY_OP("Erfc", ace::UnaryOpOperation_ERFC);
  TO_UNARY_OP("Erfinv", ace::UnaryOpOperation_ERFINV);
  TO_UNARY_OP("Expm1", ace::UnaryOpOperation_EXPM1);
  TO_UNARY_OP("Floor", ace::UnaryOpOperation_FLOOR);
  TO_UNARY_OP("HardSwish", ace::UnaryOpOperation_HARDSWISH);
  TO_UNARY_OP("Log", ace::UnaryOpOperation_LOG);
  TO_UNARY_OP("Log1p", ace::UnaryOpOperation_LOG1P);
  TO_UNARY_OP("Gelu", ace::UnaryOpOperation_GELU);
  TO_UNARY_OP("Neg", ace::UnaryOpOperation_NEG);
  TO_UNARY_OP("Sin", ace::UnaryOpOperation_SIN);
  TO_UNARY_OP("Sinh", ace::UnaryOpOperation_SINH);
  TO_UNARY_OP("Sqrt", ace::UnaryOpOperation_SQRT);
  TO_UNARY_OP("Tan", ace::UnaryOpOperation_TAN);
  TO_UNARY_OP("Tanh", ace::UnaryOpOperation_TANH);
  TO_UNARY_OP("Reciprocal", ace::UnaryOpOperation_RECIPROCAL);
  TO_UNARY_OP("Round", ace::UnaryOpOperation_ROUND);
  TO_UNARY_OP("Sign", ace::UnaryOpOperation_SIGN);

  // For specitial error onnx
  TO_UNARY_OP("ATan", ace::UnaryOpOperation_ATAN);
  dstOp->main.value = unaryOpParam.release();
}

REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Abs);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Acos);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Acosh);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Asinh);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Atan);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Atanh);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Asin);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Ceil);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Cos);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Cosh);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Expm1);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Exp);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Erf);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Erfc);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Erfinv);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Floor);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, HardSwish);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Log);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Log1p);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Gelu);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Neg);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Sin);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Tan);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Tanh);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Reciprocal);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Round);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Sign);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Sinh);
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, Sqrt);

// For specitial error onnx
REGISTER_ONNX_NODE_PARSER(UnaryOnnx, ATan);

}  // namespace parser
}  // namespace ace

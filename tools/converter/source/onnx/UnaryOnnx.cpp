//
//  UnaryOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(UnaryOnnx);

tars::OpType UnaryOnnx::opType() { return tars::OpType_UnaryOp; }

tars::OpParameter UnaryOnnx::type() { return tars::OpParameter_UnaryOp; }

void UnaryOnnx::run(tars::OpT *dstOp, const onnx::NodeProto *onnxNode,
                    OnnxScope *scope) {
  std::unique_ptr<tars::UnaryOpT> unaryOpParam(new tars::UnaryOpT);
  unaryOpParam->T = tars::DataType_DT_FLOAT;

  const auto &originalType = onnxNode->op_type();

#define TO_UNARY_OP(src, dst)   \
  if (originalType == src) {    \
    unaryOpParam->opType = dst; \
  }

  TO_UNARY_OP("Abs", tars::UnaryOpOperation_ABS);
  TO_UNARY_OP("Acos", tars::UnaryOpOperation_ACOS);
  TO_UNARY_OP("Acosh", tars::UnaryOpOperation_ACOSH);
  TO_UNARY_OP("Asinh", tars::UnaryOpOperation_ASINH);
  TO_UNARY_OP("Atan", tars::UnaryOpOperation_ATAN);
  TO_UNARY_OP("Atanh", tars::UnaryOpOperation_ATANH);
  TO_UNARY_OP("Asin", tars::UnaryOpOperation_ASIN);
  TO_UNARY_OP("Ceil", tars::UnaryOpOperation_CEIL);
  TO_UNARY_OP("Cos", tars::UnaryOpOperation_COS);
  TO_UNARY_OP("Cosh", tars::UnaryOpOperation_COSH);
  TO_UNARY_OP("Exp", tars::UnaryOpOperation_EXP);
  TO_UNARY_OP("Erf", tars::UnaryOpOperation_ERF);
  TO_UNARY_OP("Erfc", tars::UnaryOpOperation_ERFC);
  TO_UNARY_OP("Erfinv", tars::UnaryOpOperation_ERFINV);
  TO_UNARY_OP("Expm1", tars::UnaryOpOperation_EXPM1);
  TO_UNARY_OP("Floor", tars::UnaryOpOperation_FLOOR);
  TO_UNARY_OP("HardSwish", tars::UnaryOpOperation_HARDSWISH);
  TO_UNARY_OP("Log", tars::UnaryOpOperation_LOG);
  TO_UNARY_OP("Log1p", tars::UnaryOpOperation_LOG1P);
  TO_UNARY_OP("Gelu", tars::UnaryOpOperation_GELU);
  TO_UNARY_OP("Neg", tars::UnaryOpOperation_NEG);
  TO_UNARY_OP("Sin", tars::UnaryOpOperation_SIN);
  TO_UNARY_OP("Sinh", tars::UnaryOpOperation_SINH);
  TO_UNARY_OP("Sqrt", tars::UnaryOpOperation_SQRT);
  TO_UNARY_OP("Tan", tars::UnaryOpOperation_TAN);
  TO_UNARY_OP("Tanh", tars::UnaryOpOperation_TANH);
  TO_UNARY_OP("Reciprocal", tars::UnaryOpOperation_RECIPROCAL);
  TO_UNARY_OP("Round", tars::UnaryOpOperation_ROUND);
  TO_UNARY_OP("Sign", tars::UnaryOpOperation_SIGN);

  // For specitial error onnx
  TO_UNARY_OP("ATan", tars::UnaryOpOperation_ATAN);
  dstOp->main.value = unaryOpParam.release();
}

REGISTER_CONVERTER(UnaryOnnx, Abs);
REGISTER_CONVERTER(UnaryOnnx, Acos);
REGISTER_CONVERTER(UnaryOnnx, Acosh);
REGISTER_CONVERTER(UnaryOnnx, Asinh);
REGISTER_CONVERTER(UnaryOnnx, Atan);
REGISTER_CONVERTER(UnaryOnnx, Atanh);
REGISTER_CONVERTER(UnaryOnnx, Asin);
REGISTER_CONVERTER(UnaryOnnx, Ceil);
REGISTER_CONVERTER(UnaryOnnx, Cos);
REGISTER_CONVERTER(UnaryOnnx, Cosh);
REGISTER_CONVERTER(UnaryOnnx, Expm1);
REGISTER_CONVERTER(UnaryOnnx, Exp);
REGISTER_CONVERTER(UnaryOnnx, Erf);
REGISTER_CONVERTER(UnaryOnnx, Erfc);
REGISTER_CONVERTER(UnaryOnnx, Erfinv);
REGISTER_CONVERTER(UnaryOnnx, Floor);
REGISTER_CONVERTER(UnaryOnnx, HardSwish);
REGISTER_CONVERTER(UnaryOnnx, Log);
REGISTER_CONVERTER(UnaryOnnx, Log1p);
REGISTER_CONVERTER(UnaryOnnx, Gelu);
REGISTER_CONVERTER(UnaryOnnx, Neg);
REGISTER_CONVERTER(UnaryOnnx, Sin);
REGISTER_CONVERTER(UnaryOnnx, Tan);
REGISTER_CONVERTER(UnaryOnnx, Tanh);
REGISTER_CONVERTER(UnaryOnnx, Reciprocal);
REGISTER_CONVERTER(UnaryOnnx, Round);
REGISTER_CONVERTER(UnaryOnnx, Sign);
REGISTER_CONVERTER(UnaryOnnx, Sinh);
REGISTER_CONVERTER(UnaryOnnx, Sqrt);

// For specitial error onnx
REGISTER_CONVERTER(UnaryOnnx, ATan);

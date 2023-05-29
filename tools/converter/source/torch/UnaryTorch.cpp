//
//  UnaryTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(UnaryTorch);

tars::OpType UnaryTorch::opType() { return tars::OpType_UnaryOp; }

tars::OpParameter UnaryTorch::type() { return tars::OpParameter_UnaryOp; }

std::vector<int> UnaryTorch::inputTensorIdx() { return {0}; }

void UnaryTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                     TorchScope* scope) {
  static std::map<std::string, tars::UnaryOpOperation> gMaps{
      {"abs", tars::UnaryOpOperation_ABS},
      {"neg", tars::UnaryOpOperation_NEG},
      {"floor", tars::UnaryOpOperation_FLOOR},
      {"ceil", tars::UnaryOpOperation_CEIL},
      {"square", tars::UnaryOpOperation_SQUARE},
      {"sqrt", tars::UnaryOpOperation_SQRT},
      {"rsqrt", tars::UnaryOpOperation_RSQRT},
      {"exp", tars::UnaryOpOperation_EXP},
      {"log", tars::UnaryOpOperation_LOG},
      {"sin", tars::UnaryOpOperation_SIN},
      {"cos", tars::UnaryOpOperation_COS},
      {"tan", tars::UnaryOpOperation_TAN},
      {"asin", tars::UnaryOpOperation_ASIN},
      {"acos", tars::UnaryOpOperation_ACOS},
      {"atan", tars::UnaryOpOperation_ATAN},
      {"reciprocal", tars::UnaryOpOperation_RECIPROCAL},
      {"log1p", tars::UnaryOpOperation_LOG1P},
      {"bernoulli", tars::UnaryOpOperation_BNLL},
      {"acosh", tars::UnaryOpOperation_ACOSH},
      {"sinh", tars::UnaryOpOperation_SINH},
      {"asinh", tars::UnaryOpOperation_ASINH},
      {"atanh", tars::UnaryOpOperation_ATANH},
      {"sign", tars::UnaryOpOperation_SIGN},
      {"round", tars::UnaryOpOperation_ROUND},
      {"cosh", tars::UnaryOpOperation_COSH},
      {"erf", tars::UnaryOpOperation_ERF},
      {"erfc", tars::UnaryOpOperation_ERFC},
      {"erfinv", tars::UnaryOpOperation_ERFINV},
      {"expm1", tars::UnaryOpOperation_EXPM1},
      {"tanh", tars::UnaryOpOperation_TANH},
      {"sigmoid", tars::UnaryOpOperation_SIGMOID},
      {"hardswish", tars::UnaryOpOperation_HARDSWISH},
      {"gelu", tars::UnaryOpOperation_GELU_STANDARD},
  };
  auto param = new tars::UnaryOpT;
  std::string opType = getRealOpType(node);
  param->opType = gMaps[opType];
  dstOp->main.value = param;
}

REGISTER_CONVERTER(UnaryTorch, abs);
REGISTER_CONVERTER(UnaryTorch, neg);
REGISTER_CONVERTER(UnaryTorch, floor);
REGISTER_CONVERTER(UnaryTorch, ceil);
REGISTER_CONVERTER(UnaryTorch, square);
REGISTER_CONVERTER(UnaryTorch, sqrt);
REGISTER_CONVERTER(UnaryTorch, rsqrt);
REGISTER_CONVERTER(UnaryTorch, exp);
REGISTER_CONVERTER(UnaryTorch, log);
REGISTER_CONVERTER(UnaryTorch, sin);
REGISTER_CONVERTER(UnaryTorch, cos);
REGISTER_CONVERTER(UnaryTorch, tan);
REGISTER_CONVERTER(UnaryTorch, asin);
REGISTER_CONVERTER(UnaryTorch, acos);
REGISTER_CONVERTER(UnaryTorch, atan);
REGISTER_CONVERTER(UnaryTorch, reciprocal);
REGISTER_CONVERTER(UnaryTorch, log1p);
REGISTER_CONVERTER(UnaryTorch, bernoulli);
REGISTER_CONVERTER(UnaryTorch, acosh);
REGISTER_CONVERTER(UnaryTorch, sinh);
REGISTER_CONVERTER(UnaryTorch, asinh);
REGISTER_CONVERTER(UnaryTorch, atanh);
REGISTER_CONVERTER(UnaryTorch, sign);
REGISTER_CONVERTER(UnaryTorch, round);
REGISTER_CONVERTER(UnaryTorch, cosh);
REGISTER_CONVERTER(UnaryTorch, erf);
REGISTER_CONVERTER(UnaryTorch, erfc);
REGISTER_CONVERTER(UnaryTorch, erfinv);
REGISTER_CONVERTER(UnaryTorch, expm1);
REGISTER_CONVERTER(UnaryTorch, tanh);
REGISTER_CONVERTER(UnaryTorch, sigmoid);
REGISTER_CONVERTER(UnaryTorch, hardswish);
REGISTER_CONVERTER(UnaryTorch, gelu);

// TODO: silu will impl as unary ?
DECLARE_OP_CONVERTER(ExtraUnaryTorch);

tars::OpType ExtraUnaryTorch::opType() { return tars::OpType_Extra; }
tars::OpParameter ExtraUnaryTorch::type() { return tars::OpParameter_Extra; }
std::vector<int> ExtraUnaryTorch::inputTensorIdx() { return {0}; }

void ExtraUnaryTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                          TorchScope* scope) {
  auto extra = new tars::ExtraT;
  extra->engine = "Torch";
  auto type = getRealOpType(node);
  extra->type = type;
  dstOp->main.value = extra;
  if (type == "softplus") {
    extra->attr.resize(2);
    extra->attr[0].reset(new tars::AttributeT);
    extra->attr[0]->key = "beta";
    extra->attr[0]->i = getValue<int64_t>(node->input(1));
    extra->attr[1].reset(new tars::AttributeT);
    extra->attr[1]->key = "threshold";
    extra->attr[1]->i = getValue<int64_t>(node->input(2));
  }
}

REGISTER_CONVERTER(ExtraUnaryTorch, silu);
REGISTER_CONVERTER(ExtraUnaryTorch, softplus);
REGISTER_CONVERTER(ExtraUnaryTorch, bitwise_not);

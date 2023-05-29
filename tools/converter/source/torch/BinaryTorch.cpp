//
//  BinaryTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(BinaryTorch);

tars::OpType BinaryTorch::opType() { return tars::OpType_BinaryOp; }

tars::OpParameter BinaryTorch::type() { return tars::OpParameter_BinaryOp; }

std::vector<int> BinaryTorch::inputTensorIdx() { return {0, 1}; }

void BinaryTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                      TorchScope* scope) {
  static std::map<std::string, tars::BinaryOpOperation> gMaps{
      {"add", tars::BinaryOpOperation_ADD},
      {"sum", tars::BinaryOpOperation_ADD},
      {"sub", tars::BinaryOpOperation_SUB},
      {"rsub", tars::BinaryOpOperation_SUB},
      {"mul", tars::BinaryOpOperation_MUL},
      {"pow", tars::BinaryOpOperation_POW},
      {"div", tars::BinaryOpOperation_REALDIV},
      {"min_compare", tars::BinaryOpOperation_MINIMUM},
      {"minimum", tars::BinaryOpOperation_MINIMUM},
      {"max_compare", tars::BinaryOpOperation_MAXIMUM},
      {"maximum", tars::BinaryOpOperation_MAXIMUM},
      {"gt", tars::BinaryOpOperation_GREATER},
      {"greater", tars::BinaryOpOperation_GREATER},
      {"ge", tars::BinaryOpOperation_GREATER_EQUAL},
      {"lt", tars::BinaryOpOperation_LESS},
      {"less", tars::BinaryOpOperation_LESS},
      {"floordiv", tars::BinaryOpOperation_FLOORDIV},
      {"floor_divide", tars::BinaryOpOperation_FLOORDIV},
      {"le", tars::BinaryOpOperation_LESS_EQUAL},
      {"eq", tars::BinaryOpOperation_EQUAL},
      {"__is__", tars::BinaryOpOperation_EQUAL},
      {"mode", tars::BinaryOpOperation_MOD},
      {"remainder", tars::BinaryOpOperation_MOD},
      {"atan2", tars::BinaryOpOperation_ATAN2},
      {"logical_or", tars::BinaryOpOperation_LOGICALOR},
      {"__or__", tars::BinaryOpOperation_BITWISE_OR},
      {"__ior__", tars::BinaryOpOperation_BITWISE_OR},
      {"__and__", tars::BinaryOpOperation_BITWISE_AND},
      {"__iand__", tars::BinaryOpOperation_BITWISE_AND},
      {"__xor__", tars::BinaryOpOperation_BITWISE_XOR},
      {"__ixor__", tars::BinaryOpOperation_BITWISE_XOR},
      {"ne", tars::BinaryOpOperation_NOTEQUAL},
      {"__isnot__", tars::BinaryOpOperation_NOTEQUAL}};
  auto param = new tars::BinaryOpT;
  std::string opType = getRealOpType(node);
  param->opType = gMaps[opType];
  dstOp->main.value = param;
  if (opType == "rsub") {
    MNN_ASSERT(getValue<int64_t>(node->input(2)) == 1);
    int x = dstOp->inputIndexes[0];
    dstOp->inputIndexes[0] = dstOp->inputIndexes[1];
    dstOp->inputIndexes[1] = x;
  }
}

REGISTER_CONVERTER(BinaryTorch, add);
REGISTER_CONVERTER(BinaryTorch, sum_binary);
REGISTER_CONVERTER(BinaryTorch, sub);
REGISTER_CONVERTER(BinaryTorch, mul);
REGISTER_CONVERTER(BinaryTorch, pow);
REGISTER_CONVERTER(BinaryTorch, div);
REGISTER_CONVERTER(BinaryTorch, min_binary);
REGISTER_CONVERTER(BinaryTorch, minimum);
REGISTER_CONVERTER(BinaryTorch, max_binary);
REGISTER_CONVERTER(BinaryTorch, maximum);
REGISTER_CONVERTER(BinaryTorch, gt);
REGISTER_CONVERTER(BinaryTorch, greater);
REGISTER_CONVERTER(BinaryTorch, ge);
REGISTER_CONVERTER(BinaryTorch, lt);
REGISTER_CONVERTER(BinaryTorch, less);
REGISTER_CONVERTER(BinaryTorch, floordiv);
REGISTER_CONVERTER(BinaryTorch, floor_divide);
REGISTER_CONVERTER(BinaryTorch, le);
REGISTER_CONVERTER(BinaryTorch, eq);
REGISTER_CONVERTER(BinaryTorch, mode);
REGISTER_CONVERTER(BinaryTorch, remainder);
REGISTER_CONVERTER(BinaryTorch, atan2);
REGISTER_CONVERTER(BinaryTorch, logical_or);
REGISTER_CONVERTER(BinaryTorch, ne);
REGISTER_CONVERTER(BinaryTorch, rsub);
REGISTER_CONVERTER(BinaryTorch, __is__);
REGISTER_CONVERTER(BinaryTorch, __isnot__);
REGISTER_CONVERTER(BinaryTorch, __or__);
REGISTER_CONVERTER(BinaryTorch, __ior__);
REGISTER_CONVERTER(BinaryTorch, __and__);
REGISTER_CONVERTER(BinaryTorch, __iand__);
REGISTER_CONVERTER(BinaryTorch, __xor__);
REGISTER_CONVERTER(BinaryTorch, __ixor__);

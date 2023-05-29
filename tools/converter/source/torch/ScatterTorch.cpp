//
//  ScatterTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/06/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(ScatterTorch);

tars::OpType ScatterTorch::opType() { return tars::OpType_ScatterElements; }
tars::OpParameter ScatterTorch::type() { return tars::OpParameter_BinaryOp; }
std::vector<int> ScatterTorch::inputTensorIdx() { return {0, 2, 3, 1}; }

void ScatterTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                       TorchScope* scope) {
  auto param = new tars::BinaryOpT;
  if (getRealOpType(node) == "scatter_add") {
    param->opType = tars::BinaryOpOperation_ADD;
  }
  dstOp->main.value = param;
}

REGISTER_CONVERTER(ScatterTorch, scatter);
REGISTER_CONVERTER(ScatterTorch, scatter_add);

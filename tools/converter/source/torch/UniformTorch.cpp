//
//  UniformTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2022/11/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(UniformTorch);

tars::OpType UniformTorch::opType() { return tars::OpType_RandomUniform; }
tars::OpParameter UniformTorch::type() {
  return tars::OpParameter_RandomUniform;
}
std::vector<int> UniformTorch::inputTensorIdx() { return {0}; }

void UniformTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                       TorchScope* scope) {
  auto param = new tars::RandomUniformT;
  param->low = getValue<double>(node->input(1));
  param->high = getValue<double>(node->input(2));
  dstOp->main.value = param;
}

REGISTER_CONVERTER(UniformTorch, uniform);

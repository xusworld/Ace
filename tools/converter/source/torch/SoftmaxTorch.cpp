//
//  SoftmaxTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/07/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(SoftmaxTorch);

tars::OpType SoftmaxTorch::opType() { return tars::OpType_Softmax; }
tars::OpParameter SoftmaxTorch::type() { return tars::OpParameter_Axis; }
std::vector<int> SoftmaxTorch::inputTensorIdx() { return {0}; }

void SoftmaxTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                       TorchScope* scope) {
  auto param = new tars::AxisT;
  param->axis = getValue<int64_t>(node->input(1));
  dstOp->main.value = param;
}

REGISTER_CONVERTER(SoftmaxTorch, softmax);

//
//  RangeTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(RangeTorch);

tars::OpType RangeTorch::opType() { return tars::OpType_Range; }
tars::OpParameter RangeTorch::type() { return tars::OpParameter_NONE; }
std::vector<int> RangeTorch::inputTensorIdx() { return {0, 1, 2}; }

void RangeTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                     TorchScope* scope) {
  return;
}

REGISTER_CONVERTER(RangeTorch, arange);
REGISTER_CONVERTER(RangeTorch, range);

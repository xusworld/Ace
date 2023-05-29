//
//  WhereTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(WhereTorch);

tars::OpType WhereTorch::opType() { return tars::OpType_Select; }
tars::OpParameter WhereTorch::type() { return tars::OpParameter_NONE; }
std::vector<int> WhereTorch::inputTensorIdx() { return {-1}; }

void WhereTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                     TorchScope* scope) {
  return;
}

REGISTER_CONVERTER(WhereTorch, where);

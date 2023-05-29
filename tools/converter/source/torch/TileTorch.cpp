//
//  TileTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/07/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(TileTorch);

tars::OpType TileTorch::opType() { return tars::OpType_Tile; }
tars::OpParameter TileTorch::type() { return tars::OpParameter_NONE; }
std::vector<int> TileTorch::inputTensorIdx() { return {0, 1}; }

void TileTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                    TorchScope* scope) {
  return;
}

REGISTER_CONVERTER(TileTorch, repeat);

//
//  ReshapeTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(ReshapeTorch);

tars::OpType ReshapeTorch::opType() { return tars::OpType_Reshape; }
tars::OpParameter ReshapeTorch::type() { return tars::OpParameter_Reshape; }
std::vector<int> ReshapeTorch::inputTensorIdx() { return {0, 1}; }

void ReshapeTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                       TorchScope* scope) {
  auto param = new tars::ReshapeT;
  dstOp->main.value = param;
}

REGISTER_CONVERTER(ReshapeTorch, reshape);
REGISTER_CONVERTER(ReshapeTorch, view);
REGISTER_CONVERTER(ReshapeTorch, expand);

DECLARE_OP_CONVERTER(ReshapeAsTorch);

tars::OpType ReshapeAsTorch::opType() { return tars::OpType_Extra; }
tars::OpParameter ReshapeAsTorch::type() { return tars::OpParameter_Extra; }
std::vector<int> ReshapeAsTorch::inputTensorIdx() { return {0, 1}; }

void ReshapeAsTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                         TorchScope* scope) {
  auto extra = new tars::ExtraT;
  dstOp->main.value = extra;
  extra->engine = "Torch";
  extra->type = "reshape_as";
}

REGISTER_CONVERTER(ReshapeAsTorch, reshape_as);
REGISTER_CONVERTER(ReshapeAsTorch, view_as);
REGISTER_CONVERTER(ReshapeAsTorch, expand_as);

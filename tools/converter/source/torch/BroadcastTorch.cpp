//
//  BroadcastTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2022/04/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(BroadcastTorch);

tars::OpType BroadcastTorch::opType() { return tars::OpType_BroadcastTo; }
tars::OpParameter BroadcastTorch::type() { return tars::OpParameter_NONE; }
std::vector<int> BroadcastTorch::inputTensorIdx() { return {0, 1}; }

void BroadcastTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                         TorchScope* scope) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(BroadcastTorch, expand);

DECLARE_OP_CONVERTER(BroadcastAsTorch);

tars::OpType BroadcastAsTorch::opType() { return tars::OpType_Extra; }
tars::OpParameter BroadcastAsTorch::type() { return tars::OpParameter_Extra; }
std::vector<int> BroadcastAsTorch::inputTensorIdx() { return {0, 1}; }

void BroadcastAsTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                           TorchScope* scope) {
  auto extra = new tars::ExtraT;
  dstOp->main.value = extra;
  extra->engine = "Torch";
  extra->type = "broadcast_as";
}

REGISTER_CONVERTER(BroadcastAsTorch, expand_as);

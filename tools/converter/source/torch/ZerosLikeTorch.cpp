//
//  ZerosLikeTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/10/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(ZerosLikeTorch);

tars::OpType ZerosLikeTorch::opType() { return tars::OpType_ZerosLike; }
tars::OpParameter ZerosLikeTorch::type() { return tars::OpParameter_NONE; }
std::vector<int> ZerosLikeTorch::inputTensorIdx() { return {0}; }

void ZerosLikeTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                         TorchScope* scope) {
  return;
}

REGISTER_CONVERTER(ZerosLikeTorch, zeros_like);

DECLARE_OP_CONVERTER(FullLikeTorch);

tars::OpType FullLikeTorch::opType() { return tars::OpType_Extra; }
tars::OpParameter FullLikeTorch::type() { return tars::OpParameter_Extra; }
std::vector<int> FullLikeTorch::inputTensorIdx() { return {0, 1}; }

void FullLikeTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                        TorchScope* scope) {
  auto extra = new tars::ExtraT;
  extra->engine = "Torch";
  extra->type = "full_like";
  dstOp->main.value = extra;
  return;
}

REGISTER_CONVERTER(FullLikeTorch, full_like);

DECLARE_OP_CONVERTER(OnesLikeTorch);

tars::OpType OnesLikeTorch::opType() { return tars::OpType_Extra; }
tars::OpParameter OnesLikeTorch::type() { return tars::OpParameter_Extra; }
std::vector<int> OnesLikeTorch::inputTensorIdx() { return {0}; }

void OnesLikeTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                        TorchScope* scope) {
  auto extra = new tars::ExtraT;
  extra->engine = "Torch";
  extra->type = "ones_like";
  dstOp->main.value = extra;
  return;
}

REGISTER_CONVERTER(OnesLikeTorch, ones_like);

DECLARE_OP_CONVERTER(ZerosTorch);

tars::OpType ZerosTorch::opType() { return tars::OpType_Extra; }
tars::OpParameter ZerosTorch::type() { return tars::OpParameter_Extra; }
std::vector<int> ZerosTorch::inputTensorIdx() { return {0}; }

void ZerosTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                     TorchScope* scope) {
  auto extra = new tars::ExtraT;
  extra->engine = "Torch";
  extra->type = getRealOpType(node);
  dstOp->main.value = extra;
  return;
}

REGISTER_CONVERTER(ZerosTorch, zeros);
REGISTER_CONVERTER(ZerosTorch, ones);

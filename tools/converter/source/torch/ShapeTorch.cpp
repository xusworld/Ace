//
//  ShapeTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

// size -> Shape
DECLARE_OP_CONVERTER(ShapeTorch);

tars::OpType ShapeTorch::opType() { return tars::OpType_Extra; }
tars::OpParameter ShapeTorch::type() { return tars::OpParameter_Extra; }
std::vector<int> ShapeTorch::inputTensorIdx() { return {-1}; }

void ShapeTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                     TorchScope* scope) {
  auto extra = new tars::ExtraT;
  dstOp->main.value = extra;
  extra->engine = "Torch";
  extra->type = "size";
}

REGISTER_CONVERTER(ShapeTorch, size);

// dim -> Rank
DECLARE_OP_CONVERTER(RankTorch);

tars::OpType RankTorch::opType() { return tars::OpType_Rank; }
tars::OpParameter RankTorch::type() { return tars::OpParameter_NONE; }
std::vector<int> RankTorch::inputTensorIdx() { return {0}; }

void RankTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                    TorchScope* scope) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(RankTorch, dim);

// len -> Size
DECLARE_OP_CONVERTER(SizeTorch);

tars::OpType SizeTorch::opType() { return tars::OpType_Size; }
tars::OpParameter SizeTorch::type() { return tars::OpParameter_NONE; }
std::vector<int> SizeTorch::inputTensorIdx() { return {0}; }

void SizeTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                    TorchScope* scope) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(SizeTorch, len);
REGISTER_CONVERTER(SizeTorch, numel);

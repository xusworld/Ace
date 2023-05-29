//
//  IndexTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2022/04/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(IndexTorch);

tars::OpType IndexTorch::opType() { return tars::OpType_Extra; }
tars::OpParameter IndexTorch::type() { return tars::OpParameter_Extra; }
std::vector<int> IndexTorch::inputTensorIdx() { return {-1}; }

void IndexTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                     TorchScope* scope) {
  auto extra = new tars::ExtraT;
  dstOp->main.value = extra;
  extra->engine = "Torch";
  extra->type = getRealOpType(node);
}

REGISTER_CONVERTER(IndexTorch, index);
REGISTER_CONVERTER(IndexTorch, index_stridedslice);
REGISTER_CONVERTER(IndexTorch, index_put);

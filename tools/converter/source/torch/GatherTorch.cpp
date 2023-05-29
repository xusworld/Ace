//
//  GatherTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(GatherTorch);

tars::OpType GatherTorch::opType() { return tars::OpType_Gather; }
tars::OpParameter GatherTorch::type() { return tars::OpParameter_Gather; }
std::vector<int> GatherTorch::inputTensorIdx() { return {0, 1}; }

void GatherTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                      TorchScope* scope) {
  auto param = new tars::GatherT;
  std::string opType = getRealOpType(node);
  ;
  dstOp->main.value = param;
}

REGISTER_CONVERTER(GatherTorch, __getitem__);
REGISTER_CONVERTER(GatherTorch, embedding);

DECLARE_OP_CONVERTER(SelectTorch);

tars::OpType SelectTorch::opType() { return tars::OpType_Extra; }
tars::OpParameter SelectTorch::type() { return tars::OpParameter_Extra; }
std::vector<int> SelectTorch::inputTensorIdx() { return {-1}; }

void SelectTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                      TorchScope* scope) {
  auto extra = new tars::ExtraT;
  dstOp->main.value = extra;
  extra->engine = "Torch";
  extra->type = getRealOpType(node);
}

REGISTER_CONVERTER(SelectTorch, select);
REGISTER_CONVERTER(SelectTorch, index_select);

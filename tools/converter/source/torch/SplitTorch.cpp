//
//  SplitTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(SplitTorch);

tars::OpType SplitTorch::opType() { return tars::OpType_Slice; }
tars::OpParameter SplitTorch::type() { return tars::OpParameter_Slice; }
std::vector<int> SplitTorch::inputTensorIdx() { return {0}; }

void SplitTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                     TorchScope* scope) {
  auto param = new tars::SliceT;
  auto kind = node->input(1)->type()->kind();
  if (kind == c10::TypeKind::IntType) {
    param->slicePoints.push_back(getValue<int64_t>(node->input(1)));
  } else {
    auto split_sizes = getValue<std::vector<int64_t>>(node->input(1));
    for (auto i : split_sizes) {
      param->slicePoints.push_back(i);
    }
  }
  param->axis = getValue<int64_t>(node->input(2));
  // TORCH Split: param is INT: split_size; param is [INT]: [split_size_0,
  // split_size_1, ...]
  param->sourceType = tars::NetSource_TORCH;
  dstOp->main.value = param;
}

REGISTER_CONVERTER(SplitTorch, split);

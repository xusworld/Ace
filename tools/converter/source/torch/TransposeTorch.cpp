//
//  TransposeTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(PermuteTorch);

tars::OpType PermuteTorch::opType() { return tars::OpType_Permute; }
tars::OpParameter PermuteTorch::type() { return tars::OpParameter_Permute; }
std::vector<int> PermuteTorch::inputTensorIdx() { return {0}; }

void PermuteTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                       TorchScope* scope) {
  auto param = new tars::PermuteT;
  auto type = getRealOpType(node);
  if (type == "numpy_T" || type == "t") {
    param->dims = {1, 0};
  } else {
    auto dims = getValue<std::vector<int64_t>>(node->input(1));
    param->dims.resize(dims.size());
    for (int i = 0; i < dims.size(); i++) {
      param->dims[i] = dims[i];
    }
  }
  dstOp->main.value = param;
}

REGISTER_CONVERTER(PermuteTorch, permute);
REGISTER_CONVERTER(PermuteTorch, numpy_T);
REGISTER_CONVERTER(PermuteTorch, t);

DECLARE_OP_CONVERTER(TransposeTorch);

tars::OpType TransposeTorch::opType() { return tars::OpType_Extra; }
tars::OpParameter TransposeTorch::type() { return tars::OpParameter_Extra; }
std::vector<int> TransposeTorch::inputTensorIdx() { return {-1}; }

void TransposeTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                         TorchScope* scope) {
  auto extra = new tars::ExtraT;
  dstOp->main.value = extra;
  extra->engine = "Torch";
  extra->type = "transpose";
}

// aten::transpose(self : Tensor, dim0 : int , dim1 : int)
REGISTER_CONVERTER(TransposeTorch, transpose);

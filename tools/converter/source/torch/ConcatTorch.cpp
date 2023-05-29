//
//  ConcatTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(ListTorch);

tars::OpType ListTorch::opType() { return tars::OpType_Pack; }
tars::OpParameter ListTorch::type() { return tars::OpParameter_PackParam; }
std::vector<int> ListTorch::inputTensorIdx() { return {-1}; }

void ListTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                    TorchScope* scope) {
  auto param = new tars::PackParamT;
  param->axis = 0;
  if (getRealOpType(node) == "stack") {
    dstOp->inputIndexes.pop_back();
    auto axis = node->inputs().back();
    param->axis = getValue<int64_t>(axis);
  }
  dstOp->main.value = param;
}

REGISTER_CONVERTER(ListTorch, stack);
REGISTER_CONVERTER(ListTorch, ListConstruct);

DECLARE_OP_CONVERTER(TupleTorch);

tars::OpType TupleTorch::opType() { return tars::OpType_Concat; }
tars::OpParameter TupleTorch::type() { return tars::OpParameter_Axis; }
std::vector<int> TupleTorch::inputTensorIdx() { return {-1}; }

void TupleTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                     TorchScope* scope) {
  auto param = new tars::AxisT;
  param->axis = 0;
  dstOp->main.value = param;
}

REGISTER_CONVERTER(TupleTorch, TupleConstruct);

DECLARE_OP_CONVERTER(ConcatTorch);

tars::OpType ConcatTorch::opType() { return tars::OpType_Concat; }
tars::OpParameter ConcatTorch::type() { return tars::OpParameter_Axis; }
std::vector<int> ConcatTorch::inputTensorIdx() { return {}; }

void ConcatTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                      TorchScope* scope) {
  auto param = new tars::AxisT;
  const auto inputs = node->inputs();
  auto tensorlist = inputs[0];
  for (const auto input : tensorlist->node()->inputs()) {
    dstOp->inputIndexes.push_back(scope->lookupTensor(input->debugName()));
  }
  param->axis = getValue<int64_t>(inputs[1]);
  dstOp->main.value = param;
}

REGISTER_CONVERTER(ConcatTorch, cat);

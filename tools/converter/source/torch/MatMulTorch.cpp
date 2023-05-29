//
//  MatMulTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(MatMulTorch);

tars::OpType MatMulTorch::opType() { return tars::OpType_MatMul; }
tars::OpParameter MatMulTorch::type() { return tars::OpParameter_MatMul; }
std::vector<int> MatMulTorch::inputTensorIdx() { return {0, 1}; }

void MatMulTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                      TorchScope* scope) {
  auto param = new tars::MatMulT;
  std::string opType = getRealOpType(node);
  if (opType == "linear") {
    std::vector<int> shape;
    param->bias = getValue<float>(node->input(2), shape);
    param->transposeB = true;
  }
  dstOp->main.value = param;
}

REGISTER_CONVERTER(MatMulTorch, matmul);
REGISTER_CONVERTER(MatMulTorch, linear);

DECLARE_OP_CONVERTER(BatchMatMulTorch);

tars::OpType BatchMatMulTorch::opType() { return tars::OpType_BatchMatMul; }
tars::OpParameter BatchMatMulTorch::type() {
  return tars::OpParameter_BatchMatMulParam;
}
std::vector<int> BatchMatMulTorch::inputTensorIdx() { return {0, 1}; }

void BatchMatMulTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                           TorchScope* scope) {
  auto param = new tars::BatchMatMulParamT;
  dstOp->main.value = param;
}

REGISTER_CONVERTER(BatchMatMulTorch, bmm);

DECLARE_OP_CONVERTER(AddmmTorch);

tars::OpType AddmmTorch::opType() { return tars::OpType_Extra; }
tars::OpParameter AddmmTorch::type() { return tars::OpParameter_Extra; }
std::vector<int> AddmmTorch::inputTensorIdx() { return {0, 1, 2}; }

void AddmmTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                     TorchScope* scope) {
  auto extra = new tars::ExtraT;
  dstOp->main.value = extra;
  extra->engine = "Torch";
  extra->type = getRealOpType(node);
  const auto inputs = node->inputs();
  const auto beta = inputs[3];
  const auto alpha = inputs[4];
  extra->attr.resize(2);
  extra->attr[0].reset(new tars::AttributeT);
  extra->attr[0]->key = "beta";
  extra->attr[0]->i = getValue<int64_t>(beta);
  extra->attr[1].reset(new tars::AttributeT);
  extra->attr[1]->key = "alpha";
  extra->attr[1]->i = getValue<int64_t>(alpha);
}

REGISTER_CONVERTER(AddmmTorch, addmm);

DECLARE_OP_CONVERTER(EinsumTorch);

tars::OpType EinsumTorch::opType() { return tars::OpType_Extra; }
tars::OpParameter EinsumTorch::type() { return tars::OpParameter_Extra; }
std::vector<int> EinsumTorch::inputTensorIdx() { return {1}; }

void EinsumTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                      TorchScope* scope) {
  auto extra = new tars::ExtraT;
  dstOp->main.value = extra;
  extra->engine = "Torch";
  extra->type = getRealOpType(node);
  const auto inputs = node->inputs();
  const auto beta = inputs[3];
  const auto alpha = inputs[4];
  extra->attr.resize(2);
  extra->attr[0].reset(new tars::AttributeT);
  extra->attr[0]->key = "beta";
  extra->attr[0]->i = getValue<int64_t>(beta);
  extra->attr[1].reset(new tars::AttributeT);
  extra->attr[1]->key = "alpha";
  extra->attr[1]->i = getValue<int64_t>(alpha);
}

REGISTER_CONVERTER(EinsumTorch, einsum);

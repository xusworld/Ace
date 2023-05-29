//
//  SqueezeTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(UnSqueezeTorch);

tars::OpType UnSqueezeTorch::opType() { return tars::OpType_Unsqueeze; }
tars::OpParameter UnSqueezeTorch::type() {
  return tars::OpParameter_SqueezeParam;
}
std::vector<int> UnSqueezeTorch::inputTensorIdx() { return {0}; }

void UnSqueezeTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                         TorchScope* scope) {
  auto param = new tars::SqueezeParamT;
  if (node->inputs().size() > 1) {
    param->squeezeDims.push_back(getValue<int64_t>(node->input(1)));
  }
  dstOp->main.value = param;
}

REGISTER_CONVERTER(UnSqueezeTorch, unsqueeze);

DECLARE_OP_CONVERTER(SqueezeTorch);

tars::OpType SqueezeTorch::opType() { return tars::OpType_Squeeze; }
tars::OpParameter SqueezeTorch::type() {
  return tars::OpParameter_SqueezeParam;
}
std::vector<int> SqueezeTorch::inputTensorIdx() { return {0}; }

void SqueezeTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                       TorchScope* scope) {
  auto param = new tars::SqueezeParamT;
  if (node->inputs().size() > 1) {
    param->squeezeDims.push_back(getValue<int64_t>(node->input(1)));
  }
  dstOp->main.value = param;
}

REGISTER_CONVERTER(SqueezeTorch, squeeze);

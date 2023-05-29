//
//  BatchNormTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(BatchNormTorch);

tars::OpType BatchNormTorch::opType() { return tars::OpType_BatchNorm; }
tars::OpParameter BatchNormTorch::type() { return tars::OpParameter_BatchNorm; }
std::vector<int> BatchNormTorch::inputTensorIdx() { return {0}; }

void BatchNormTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                         TorchScope* scope) {
  auto param = new tars::BatchNormT;
  const auto& inputs = node->inputs();
  const auto slope = inputs[1];
  const auto bias = inputs[2];
  const auto mean = inputs[3];
  const auto var = inputs[4];
  const auto epsilon = inputs[7];
  std::vector<int> shape;
  param->slopeData = getValue<float>(slope, shape);
  param->channels = shape[0];
  param->biasData = getValue<float>(bias, shape);
  param->meanData = getValue<float>(mean, shape);
  param->varData = getValue<float>(var, shape);
  param->epsilon = getValue<float>(epsilon);
  param->Adata = std::vector<float>(param->channels, 0.f);
  param->Bdata = std::vector<float>(param->channels, 0.f);
  dstOp->main.value = param;
}

REGISTER_CONVERTER(BatchNormTorch, batch_norm);

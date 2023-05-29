//
//  CumPordTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2022/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(CumPordTorch);

tars::OpType CumPordTorch::opType() { return tars::OpType_CumProd; }
tars::OpParameter CumPordTorch::type() { return tars::OpParameter_Axis; }
std::vector<int> CumPordTorch::inputTensorIdx() { return {0}; }

void CumPordTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                       TorchScope* scope) {
  auto param = new tars::AxisT;
  param->axis = getValue<int64_t>(node->input(1));
  dstOp->main.value = param;
}

REGISTER_CONVERTER(CumPordTorch, cumprod);

//
//  SpaceToDepthTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/06/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(DepthToSpaceTorch);

tars::OpType DepthToSpaceTorch::opType() { return tars::OpType_DepthToSpace; }

tars::OpParameter DepthToSpaceTorch::type() {
  return tars::OpParameter_DepthSpaceParam;
}

std::vector<int> DepthToSpaceTorch::inputTensorIdx() { return {0}; }

void DepthToSpaceTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                            TorchScope* scope) {
  auto param = new tars::DepthSpaceParamT;
  std::string opType = getRealOpType(node);
  const auto upscale = node->inputs()[1];
  param->blockSize = getValue<int64_t>(upscale);
  dstOp->main.value = param;
}

REGISTER_CONVERTER(DepthToSpaceTorch, pixel_shuffle);

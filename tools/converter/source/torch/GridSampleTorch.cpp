//
//  GridSampleTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2022/04/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(GridSampleTorch);

tars::OpType GridSampleTorch::opType() { return tars::OpType_GridSample; }
tars::OpParameter GridSampleTorch::type() {
  return tars::OpParameter_GridSample;
}
std::vector<int> GridSampleTorch::inputTensorIdx() { return {0, 1}; }

void GridSampleTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                          TorchScope* scope) {
  auto gridSampleParam = new tars::GridSampleT;
  int mode = getValue<int64_t>(node->input(2));
  if (mode == 0 || mode == 1) {
    gridSampleParam->mode = static_cast<tars::SampleMode>(mode);
  } else {
    LOG(FATAL) << "Unknown mode for " << dstOp->name << "!";
  }
  int padding_mode = getValue<int64_t>(node->input(3));
  if (padding_mode == 0 || padding_mode == 1 || padding_mode == 2) {
    gridSampleParam->paddingMode = static_cast<tars::BorderMode>(mode);
  } else {
    LOG(FATAL) << "Unknown padding for " << dstOp->name << "!";
  }
  gridSampleParam->alignCorners = getValue<bool>(node->input(4));
  dstOp->main.value = gridSampleParam;
}

REGISTER_CONVERTER(GridSampleTorch, grid_sampler);

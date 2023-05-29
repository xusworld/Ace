//
//  GatherTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(GatherTflite);
tars::OpType GatherTflite::opType(bool quantizedModel) {
  return tars::OpType_Gather;
}
tars::OpParameter GatherTflite::type(bool quantizedModel) {
  return tars::OpParameter_Gather;
}

void GatherTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  auto parameter = new tars::GatherT;
  auto opt = tfliteOp->builtin_options.AsGatherOptions();
  parameter->axis = opt->axis;
  dstOp->main.value = parameter;
}

using namespace tflite;
REGISTER_CONVERTER(GatherTflite, BuiltinOperator_GATHER);

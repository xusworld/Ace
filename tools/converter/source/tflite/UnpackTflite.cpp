//
//  UnpackTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"
using namespace tflite;

DECLARE_OP_COVERTER(UnpackTflite);

tars::OpType UnpackTflite::opType(bool quantizedModel) {
  return tars::OpType_Unpack;
}
tars::OpParameter UnpackTflite::type(bool quantizedModel) {
  return tars::OpParameter_Axis;
}

void UnpackTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  auto axisT = new tars::AxisT;
  auto opt = tfliteOp->builtin_options.AsUnpackOptions();
  axisT->axis = opt->axis;
  dstOp->main.value = axisT;
}

REGISTER_CONVERTER(UnpackTflite, BuiltinOperator_UNPACK);

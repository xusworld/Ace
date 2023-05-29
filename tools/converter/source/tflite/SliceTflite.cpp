//
//  SliceTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2021/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(SliceTflite);
tars::OpType SliceTflite::opType(bool quantizedModel) {
  return tars::OpType_SliceTf;
}
tars::OpParameter SliceTflite::type(bool quantizedModel) {
  return tars::OpParameter_NONE;
}

void SliceTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  // Do nothing
}

using namespace tflite;
REGISTER_CONVERTER(SliceTflite, BuiltinOperator_SLICE);

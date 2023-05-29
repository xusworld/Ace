//
//  SelectTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2021/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(SelectTflite);
tars::OpType SelectTflite::opType(bool quantizedModel) {
  return tars::OpType_Select;
}
tars::OpParameter SelectTflite::type(bool quantizedModel) {
  return tars::OpParameter_NONE;
}

void SelectTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  // Do nothing
}

using namespace tflite;
REGISTER_CONVERTER(SelectTflite, BuiltinOperator_SELECT);

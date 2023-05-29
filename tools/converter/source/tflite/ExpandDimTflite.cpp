//
//  ExpandDimTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2021/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(ExpandDimTflite);
tars::OpType ExpandDimTflite::opType(bool quantizedModel) {
  return tars::OpType_ExpandDims;
}
tars::OpParameter ExpandDimTflite::type(bool quantizedModel) {
  return tars::OpParameter_ExpandDims;
}

void ExpandDimTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  dstOp->main.value = new tars::ExpandDimsT;
}

using namespace tflite;
REGISTER_CONVERTER(ExpandDimTflite, BuiltinOperator_EXPAND_DIMS);

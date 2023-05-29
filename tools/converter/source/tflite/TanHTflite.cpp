//
//  TanHTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(TanHTflite);
tars::OpType TanHTflite::opType(bool quantizedModel) {
  return tars::OpType_TanH;
}
tars::OpParameter TanHTflite::type(bool quantizedModel) {
  return tars::OpParameter_NONE;
}

void TanHTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  dstOp->main.value = nullptr;
}

using namespace tflite;
REGISTER_CONVERTER(TanHTflite, BuiltinOperator_TANH);

//
//  TileTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(TileTflite);
tars::OpType TileTflite::opType(bool quantizedModel) {
  return tars::OpType_Tile;
}
tars::OpParameter TileTflite::type(bool quantizedModel) {
  return tars::OpParameter_NONE;
}

void TileTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  dstOp->main.value = nullptr;
}

using namespace tflite;
REGISTER_CONVERTER(TileTflite, BuiltinOperator_TILE);

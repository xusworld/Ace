//
//  FillTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(FillTflite);

tars::OpType FillTflite::opType(bool quantizedModel) {
  return tars::OpType_Fill;
}
tars::OpParameter FillTflite::type(bool quantizedModel) {
  return tars::OpParameter_Fill;
}

void FillTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  dstOp->main.value = nullptr;
}
DECLARE_OP_COVERTER(ZerosLikeTflite);
tars::OpType ZerosLikeTflite::opType(bool quantizedModel) {
  return tars::OpType_ZerosLike;
}
tars::OpParameter ZerosLikeTflite::type(bool quantizedModel) {
  return tars::OpParameter_NONE;
}

void ZerosLikeTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  dstOp->main.value = nullptr;
}
using namespace tflite;
REGISTER_CONVERTER(FillTflite, BuiltinOperator_FILL);
REGISTER_CONVERTER(ZerosLikeTflite, BuiltinOperator_ZEROS_LIKE);

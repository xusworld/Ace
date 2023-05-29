//
//  BatchToSpaceTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2021/04/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(BatchToSpaceTflite);
tars::OpType BatchToSpaceTflite::opType(bool quantizedModel) {
  return tars::OpType_Extra;
}
tars::OpParameter BatchToSpaceTflite::type(bool quantizedModel) {
  return tars::OpParameter_Extra;
}

void BatchToSpaceTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  auto extraOpParam = new tars::ExtraT;
  extraOpParam->engine = "Tflite";
  extraOpParam->type = "BatchToSpace";
  dstOp->main.value = extraOpParam;
}

DECLARE_OP_COVERTER(SpaceToBatchTflite);
tars::OpType SpaceToBatchTflite::opType(bool quantizedModel) {
  return tars::OpType_Extra;
}
tars::OpParameter SpaceToBatchTflite::type(bool quantizedModel) {
  return tars::OpParameter_Extra;
}

void SpaceToBatchTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  auto extraOpParam = new tars::ExtraT;
  extraOpParam->engine = "Tflite";
  extraOpParam->type = "SpaceToBatch";
  dstOp->main.value = extraOpParam;
}

using namespace tflite;
REGISTER_CONVERTER(BatchToSpaceTflite, BuiltinOperator_BATCH_TO_SPACE_ND);
REGISTER_CONVERTER(SpaceToBatchTflite, BuiltinOperator_SPACE_TO_BATCH_ND);

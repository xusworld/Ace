//
//  ReshapeTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(ReshapeTflite);

tars::OpType ReshapeTflite::opType(bool quantizedModel) {
  if (quantizedModel) return tars::OpType_QuantizedReshape;
  return tars::OpType_Reshape;
}
tars::OpParameter ReshapeTflite::type(bool quantizedModel) {
  return tars::OpParameter_Reshape;
}

void ReshapeTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  auto reshapeParam = new tars::ReshapeT;
  reshapeParam->dimType = tars::MNN_DATA_FORMAT_NHWC;

  dstOp->main.value = reshapeParam;
  auto reshape = tfliteOp->builtin_options.AsReshapeOptions();
  if (nullptr != reshape) {
    reshapeParam->dims = reshape->new_shape;
  }
}

using namespace tflite;
REGISTER_CONVERTER(ReshapeTflite, BuiltinOperator_RESHAPE);

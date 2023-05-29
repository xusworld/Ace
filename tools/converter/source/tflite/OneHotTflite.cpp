//
//  OneHotTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"
using namespace tflite;

DECLARE_OP_COVERTER(OneHotTflite);

tars::OpType OneHotTflite::opType(bool quantizedModel) {
  return tars::OpType_OneHot;
}
tars::OpParameter OneHotTflite::type(bool quantizedModel) {
  return tars::OpParameter_OneHotParam;
}

void OneHotTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  auto ohParam = new tars::OneHotParamT;
  auto opt = tfliteOp->builtin_options.AsOneHotOptions();
  ohParam->axis = opt->axis;
  dstOp->main.value = ohParam;
}

REGISTER_CONVERTER(OneHotTflite, BuiltinOperator_ONE_HOT);

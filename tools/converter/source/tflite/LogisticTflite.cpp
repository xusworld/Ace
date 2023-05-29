//
//  LogisticTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(LogisticTflite);

tars::OpType LogisticTflite::opType(bool quantizedModel) {
  if (quantizedModel) return tars::OpType_QuantizedLogistic;
  return tars::OpType_Sigmoid;
}
tars::OpParameter LogisticTflite::type(bool quantizedModel) {
  if (quantizedModel) return tars::OpParameter_QuantizedLogistic;
  return tars::OpParameter_NONE;
}

void LogisticTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  if (quantizedModel) {
    auto LogisticParam = new tars::QuantizedLogisticT;

    const int inputIndex = tfliteOp->inputs[0];
    const auto& inputTensor = tfliteTensors[inputIndex];
    LogisticParam->inputQuantizedParam =
        std::unique_ptr<tars::QuantizedParamT>(new tars::QuantizedParamT);
    LogisticParam->inputQuantizedParam->zeroPoint =
        inputTensor->quantization->zero_point[0];
    LogisticParam->inputQuantizedParam->scale =
        inputTensor->quantization->scale[0];

    const int outputIndex = tfliteOp->outputs[0];
    const auto& outputTensor = tfliteTensors[outputIndex];
    LogisticParam->outputQuantizedParam =
        std::unique_ptr<tars::QuantizedParamT>(new tars::QuantizedParamT);
    LogisticParam->outputQuantizedParam->zeroPoint =
        outputTensor->quantization->zero_point[0];
    LogisticParam->outputQuantizedParam->scale =
        outputTensor->quantization->scale[0];

    dstOp->main.value = LogisticParam;
  } else {
    dstOp->main.value = nullptr;
  }
}

using namespace tflite;
REGISTER_CONVERTER(LogisticTflite, BuiltinOperator_LOGISTIC);

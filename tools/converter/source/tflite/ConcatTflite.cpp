//
//  ConcatTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(ConcatTflite);
tars::OpType ConcatTflite::opType(bool quantizedModel) {
  if (quantizedModel) return tars::OpType_QuantizedConcat;
  return tars::OpType_Concat;
}
tars::OpParameter ConcatTflite::type(bool quantizedModel) {
  if (quantizedModel) return tars::OpParameter_QuantizedConcat;
  return tars::OpParameter_Axis;
}

void ConcatTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  const auto& tfliteConcatOption =
      tfliteOp->builtin_options.AsConcatenationOptions();
  if (quantizedModel) {
    auto concatParamQuan = new tars::QuantizedConcatT;
    concatParamQuan->axis = tfliteConcatOption->axis;

    for (int i = 0; i < tfliteOp->inputs.size(); i++) {
      const int inputIndex = tfliteOp->inputs[i];
      const auto& inputTensor = tfliteTensors[inputIndex];
      auto quantized_param_ptr =
          std::unique_ptr<tars::QuantizedParamT>(new tars::QuantizedParamT);
      concatParamQuan->inputZeroPoint.push_back(
          inputTensor->quantization->zero_point[0]);
      concatParamQuan->inputScale.push_back(
          inputTensor->quantization->scale[0]);
    }

    const int outputIndex = tfliteOp->outputs[0];
    const auto& outputTensor = tfliteTensors[outputIndex];
    concatParamQuan->outputQuantizedParam =
        std::unique_ptr<tars::QuantizedParamT>(new tars::QuantizedParamT);
    concatParamQuan->outputQuantizedParam->zeroPoint =
        outputTensor->quantization->zero_point[0];
    concatParamQuan->outputQuantizedParam->scale =
        outputTensor->quantization->scale[0];
    concatParamQuan->activationType = static_cast<tars::FusedActivation>(
        tfliteConcatOption->fused_activation_function);
    dstOp->main.value = concatParamQuan;
  } else {
    DCHECK(tfliteConcatOption->fused_activation_function ==
           tflite::ActivationFunctionType_NONE);
    auto concatParamFloat = new tars::AxisT;
    concatParamFloat->axis = tfliteConcatOption->axis;
    dstOp->main.value = concatParamFloat;
  }
}

using namespace tflite;
REGISTER_CONVERTER(ConcatTflite, BuiltinOperator_CONCATENATION);

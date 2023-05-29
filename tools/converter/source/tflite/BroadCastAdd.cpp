//
//  BroadCastAdd.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(AddTflite);

tars::OpType AddTflite::opType(bool quantizedModel) {
  if (quantizedModel) return tars::OpType_QuantizedAdd;
  return tars::OpType_Extra;
}

tars::OpParameter AddTflite::type(bool quantizedModel) {
  if (quantizedModel) return tars::OpParameter_QuantizedAdd;
  return tars::OpParameter_Extra;
}

void AddTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  const auto& addOption = tfliteOp->builtin_options.AsAddOptions();
  if (quantizedModel) {
    auto AddParam = new tars::QuantizedAddT;

    DCHECK(tfliteOp->inputs.size() == 2) << "tflite Reshape input ERROR";

    // input1
    const int input1Index = tfliteOp->inputs[0];
    const auto& input1Tensor = tfliteTensors[input1Index];
    AddParam->input1QuantizedParam =
        std::unique_ptr<tars::QuantizedParamT>(new tars::QuantizedParamT);
    AddParam->input1QuantizedParam->zeroPoint =
        input1Tensor->quantization->zero_point[0];
    AddParam->input1QuantizedParam->scale =
        input1Tensor->quantization->scale[0];

    // input1
    const int input2Index = tfliteOp->inputs[1];
    const auto& input2Tensor = tfliteTensors[input2Index];
    AddParam->input2QuantizedParam =
        std::unique_ptr<tars::QuantizedParamT>(new tars::QuantizedParamT);
    AddParam->input2QuantizedParam->zeroPoint =
        input2Tensor->quantization->zero_point[0];
    AddParam->input2QuantizedParam->scale =
        input2Tensor->quantization->scale[0];

    // output
    const int outputIndex = tfliteOp->outputs[0];
    const auto& outputTensor = tfliteTensors[outputIndex];
    AddParam->outputQuantizedParam =
        std::unique_ptr<tars::QuantizedParamT>(new tars::QuantizedParamT);
    AddParam->outputQuantizedParam->zeroPoint =
        outputTensor->quantization->zero_point[0];
    AddParam->outputQuantizedParam->scale =
        outputTensor->quantization->scale[0];

    AddParam->activationType = static_cast<tars::FusedActivation>(
        addOption->fused_activation_function);

    dstOp->main.value = AddParam;
  } else {
    auto extraOpParam = new tars::ExtraT;
    extraOpParam->engine = "Tflite";
    extraOpParam->type = "BinaryActivation";
    extraOpParam->attr.resize(2);
    extraOpParam->attr[0].reset(new tars::AttributeT);
    extraOpParam->attr[1].reset(new tars::AttributeT);
    extraOpParam->attr[0]->key = "opType";
    extraOpParam->attr[0]->i = tflite::BuiltinOperator_ADD;
    extraOpParam->attr[1]->key = "activationType";
    extraOpParam->attr[1]->i = addOption->fused_activation_function;
    dstOp->main.value = extraOpParam;
  }
}

using namespace tflite;
REGISTER_CONVERTER(AddTflite, BuiltinOperator_ADD);

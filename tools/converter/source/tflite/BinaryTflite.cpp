//
//  BinaryTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

using namespace tflite;

DECLARE_OP_COVERTER(BinaryTflite);

tars::OpType BinaryTflite::opType(bool quantizedModel) {
  return tars::OpType_Extra;
}
tars::OpParameter BinaryTflite::type(bool quantizedModel) {
  return tars::OpParameter_Extra;
}

void BinaryTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  auto extraOpParam = new tars::ExtraT;
  extraOpParam->engine = "Tflite";
  extraOpParam->type = "BinaryActivation";
  extraOpParam->attr.resize(2);
  extraOpParam->attr[0].reset(new tars::AttributeT);
  extraOpParam->attr[1].reset(new tars::AttributeT);
  extraOpParam->attr[0]->key = "opType";
  extraOpParam->attr[0]->i = tfliteOpSet[tfliteOp->opcode_index]->builtin_code;
  extraOpParam->attr[1]->key = "activationType";
  const auto& addOption = tfliteOp->builtin_options.AsAddOptions();
  if (addOption && addOption->fused_activation_function) {
    extraOpParam->attr[1]->i = addOption->fused_activation_function;
  } else {
    extraOpParam->attr[1]->i = tflite::ActivationFunctionType_NONE;
  }
  dstOp->main.value = extraOpParam;
}
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_POW);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_MAXIMUM);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_MINIMUM);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_LESS);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_GREATER_EQUAL);
// REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_ADD);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_SUB);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_FLOOR_DIV);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_FLOOR_MOD);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_LESS_EQUAL);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_GREATER);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_EQUAL);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_NOT_EQUAL);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_SQUARED_DIFFERENCE);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_MUL);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_LOGICAL_AND);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_DIV);

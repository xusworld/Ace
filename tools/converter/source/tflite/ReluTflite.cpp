//
//  ReluTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(ReluTflite);
tars::OpType ReluTflite::opType(bool quantizedModel) {
  return tars::OpType_ReLU;
}
tars::OpParameter ReluTflite::type(bool quantizedModel) {
  return tars::OpParameter_Relu;
}

void ReluTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  auto Relu = new tars::ReluT;
  Relu->slope = 0.0f;
  dstOp->main.value = Relu;
}

DECLARE_OP_COVERTER(LeakyReluTflite);
tars::OpType LeakyReluTflite::opType(bool quantizedModel) {
  return tars::OpType_ReLU;
}
tars::OpParameter LeakyReluTflite::type(bool quantizedModel) {
  return tars::OpParameter_Relu;
}

void LeakyReluTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  auto Relu = new tars::ReluT;
  auto opt = tfliteOp->builtin_options.AsLeakyReluOptions();
  Relu->slope = opt->alpha;
  dstOp->main.value = Relu;
}

DECLARE_OP_COVERTER(Relu6Tflite);
tars::OpType Relu6Tflite::opType(bool quantizedModel) {
  return tars::OpType_ReLU6;
}
tars::OpParameter Relu6Tflite::type(bool quantizedModel) {
  return tars::OpParameter_Relu6;
}

void Relu6Tflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  auto relu6 = new tars::Relu6T;
  dstOp->main.value = relu6;
}
DECLARE_OP_COVERTER(PreluTflite);
tars::OpType PreluTflite::opType(bool quantizedModel) {
  return tars::OpType_Extra;
}
tars::OpParameter PreluTflite::type(bool quantizedModel) {
  return tars::OpParameter_Extra;
}

void PreluTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  dstOp->main.value = new tars::ExtraT;
  auto dstP = dstOp->main.AsExtra();
  dstP->engine = "Tflite";
  dstP->type = "PRELU";
}

using namespace tflite;
REGISTER_CONVERTER(ReluTflite, BuiltinOperator_RELU);
REGISTER_CONVERTER(LeakyReluTflite, BuiltinOperator_LEAKY_RELU);
REGISTER_CONVERTER(Relu6Tflite, BuiltinOperator_RELU6);
REGISTER_CONVERTER(PreluTflite, BuiltinOperator_PRELU);

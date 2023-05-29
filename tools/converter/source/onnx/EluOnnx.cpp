//
//  EluOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(EluOnnx);
DECLARE_OP_CONVERTER(SEluOnnx);

tars::OpType EluOnnx::opType() { return tars::OpType_ELU; }
tars::OpType SEluOnnx::opType() { return tars::OpType_Selu; }

tars::OpParameter EluOnnx::type() { return tars::OpParameter_ELU; }
tars::OpParameter SEluOnnx::type() { return tars::OpParameter_Selu; }

void EluOnnx::run(tars::OpT *dstOp, const onnx::NodeProto *onnxNode,
                  OnnxScope *scope) {
  auto eluParam = new tars::ELUT;

  float alpha = 1.0f;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "alpha") {
      alpha = attributeProto.f();
    }
  }

  eluParam->alpha = alpha;

  dstOp->main.value = eluParam;
}
void SEluOnnx::run(tars::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope *scope) {
  auto seluParam = new tars::SeluT;

  float alpha = 1.67326, gamma = 1.0507;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "alpha") {
      alpha = attributeProto.f();
    } else if (attributeName == "gamma") {
      gamma = attributeProto.f();
    }
  }

  seluParam->alpha = alpha;
  seluParam->scale = gamma;

  dstOp->main.value = seluParam;
}

REGISTER_CONVERTER(EluOnnx, Elu);
REGISTER_CONVERTER(SEluOnnx, Selu);

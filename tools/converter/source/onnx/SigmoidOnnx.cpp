//
//  SigmoidOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(SigmoidOnnx);

tars::OpType SigmoidOnnx::opType() { return tars::OpType_Sigmoid; }

tars::OpParameter SigmoidOnnx::type() { return tars::OpParameter_NONE; }

void SigmoidOnnx::run(tars::OpT *dstOp, const onnx::NodeProto *onnxNode,
                      OnnxScope *scope) {
  return;
}

REGISTER_CONVERTER(SigmoidOnnx, Sigmoid);

//
//  IdentityOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(IdentityOnnx);

tars::OpType IdentityOnnx::opType() { return tars::OpType_Identity; }
tars::OpParameter IdentityOnnx::type() { return tars::OpParameter_NONE; }

void IdentityOnnx::run(tars::OpT *dstOp, const onnx::NodeProto *onnxNode,
                       OnnxScope *scope) {
  // Do nothing
  return;
}

REGISTER_CONVERTER(IdentityOnnx, Identity);

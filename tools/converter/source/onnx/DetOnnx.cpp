//
//  DetOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(DetOnnx);

tars::OpType DetOnnx::opType() { return tars::OpType_Det; }
tars::OpParameter DetOnnx::type() { return tars::OpParameter_NONE; }

void DetOnnx::run(tars::OpT *dstOp, const onnx::NodeProto *onnxNode,
                  OnnxScope *scope) {}

REGISTER_CONVERTER(DetOnnx, Det);

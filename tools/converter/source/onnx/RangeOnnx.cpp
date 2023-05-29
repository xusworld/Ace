//
//  RangeOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(RangeOnnx);

tars::OpType RangeOnnx::opType() { return tars::OpType_Range; }

tars::OpParameter RangeOnnx::type() { return tars::OpParameter_NONE; }

void RangeOnnx::run(tars::OpT *dstOp, const onnx::NodeProto *onnxNode,
                    OnnxScope *scope) {
  return;
}

REGISTER_CONVERTER(RangeOnnx, Range);

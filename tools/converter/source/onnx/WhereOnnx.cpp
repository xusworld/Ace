//
//  WhereOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(WhereOnnx);

tars::OpType WhereOnnx::opType() { return tars::OpType_Select; }

tars::OpParameter WhereOnnx::type() { return tars::OpParameter_NONE; }

void WhereOnnx::run(tars::OpT *dstOp, const onnx::NodeProto *onnxNode,
                    OnnxScope *scope) {
  return;
}

REGISTER_CONVERTER(WhereOnnx, Where);

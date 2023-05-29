//
//  TanhOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(TanhOnnx);

tars::OpType TanhOnnx::opType() { return tars::OpType_TanH; }
tars::OpParameter TanhOnnx::type() { return tars::OpParameter_NONE; }

void TanhOnnx::run(tars::OpT* dstOp, const onnx::NodeProto* onnxNode,
                   OnnxScope* scope) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(TanhOnnx, Tanh);

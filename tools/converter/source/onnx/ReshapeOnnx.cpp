//
//  ReshapeOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ReshapeOnnx);

tars::OpType ReshapeOnnx::opType() { return tars::OpType_Reshape; }
tars::OpParameter ReshapeOnnx::type() { return tars::OpParameter_Reshape; }

void ReshapeOnnx::run(tars::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      OnnxScope* scope) {
  auto para = new tars::ReshapeT;
  para->dimType = tars::MNN_DATA_FORMAT_NCHW;
  dstOp->main.value = para;
}

REGISTER_CONVERTER(ReshapeOnnx, Reshape);

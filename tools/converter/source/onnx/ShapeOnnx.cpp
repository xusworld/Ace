//
//  ShapeOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ShapeOnnx);

tars::OpType ShapeOnnx::opType() { return tars::OpType_Shape; }
tars::OpParameter ShapeOnnx::type() { return tars::OpParameter_NONE; }

void ShapeOnnx::run(tars::OpT* dstOp, const onnx::NodeProto* onnxNode,
                    OnnxScope* scope) {
  dstOp->defaultDimentionFormat = tars::MNN_DATA_FORMAT_NCHW;
}

REGISTER_CONVERTER(ShapeOnnx, Shape);

DECLARE_OP_CONVERTER(SizeOnnx);

tars::OpType SizeOnnx::opType() { return tars::OpType_Size; }
tars::OpParameter SizeOnnx::type() { return tars::OpParameter_NONE; }

void SizeOnnx::run(tars::OpT* dstOp, const onnx::NodeProto* onnxNode,
                   OnnxScope* scope) {
  dstOp->defaultDimentionFormat = tars::MNN_DATA_FORMAT_NCHW;
}

REGISTER_CONVERTER(SizeOnnx, Size);

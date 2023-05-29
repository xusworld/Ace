//
//  TileOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(TileOnnx);

tars::OpType TileOnnx::opType() { return tars::OpType_Tile; }

tars::OpParameter TileOnnx::type() { return tars::OpParameter_NONE; }

void TileOnnx::run(tars::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope *scope) {
  return;
}

REGISTER_CONVERTER(TileOnnx, Tile);

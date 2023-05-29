//
//  EyeLikeOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/05/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(EyeLikeOnnx);

tars::OpType EyeLikeOnnx::opType() { return tars::OpType_EyeLike; }
tars::OpParameter EyeLikeOnnx::type() { return tars::OpParameter_NONE; }

void EyeLikeOnnx::run(tars::OpT *dstOp, const onnx::NodeProto *onnxNode,
                      OnnxScope *scope) {
  int k = 0;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attr = onnxNode->attribute(i);
    const auto &attrName = attr.name();
    if (attrName == "k") {
      k = attr.i();
    }
  }
  dstOp->inputIndexes.push_back(
      scope->buildIntConstOp({k}, dstOp->name + "/k"));
}

REGISTER_CONVERTER(EyeLikeOnnx, EyeLike);

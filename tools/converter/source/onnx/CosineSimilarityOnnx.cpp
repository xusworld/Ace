//
//  CosineSimilarityOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(CosineSimilarityOnnx);

tars::OpType CosineSimilarityOnnx::opType() {
  return tars::OpType_CosineSimilarity;
}

tars::OpParameter CosineSimilarityOnnx::type() {
  return tars::OpParameter_NONE;
}

void CosineSimilarityOnnx::run(tars::OpT *dstOp,
                               const onnx::NodeProto *onnxNode,
                               OnnxScope *scope) {
  std::string type;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    auto att = onnxNode->attribute(i);
    if ("operator" == att.name()) {
      type = att.s();
      break;
    }
  }
  DCHECK(type == "cosine_similarity") << " NOT SUPPPRT";
  return;
}

REGISTER_CONVERTER(CosineSimilarityOnnx, ATen);

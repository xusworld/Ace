//
//  TransposeOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(TransposeOnnx);

tars::OpType TransposeOnnx::opType() { return tars::OpType_Permute; }

tars::OpParameter TransposeOnnx::type() { return tars::OpParameter_Permute; }

void TransposeOnnx::run(tars::OpT *dstOp, const onnx::NodeProto *onnxNode,
                        OnnxScope *scope) {
  auto param = new tars::PermuteT;

  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "perm") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS)
          << "Node Attribute ERROR";
      param->dims.resize(attributeProto.ints_size());
      for (int v = 0; v < attributeProto.ints_size(); ++v) {
        param->dims[v] = attributeProto.ints(v);
      }
    }
  }
  dstOp->main.value = param;
}

REGISTER_CONVERTER(TransposeOnnx, Transpose);

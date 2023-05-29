//
//  SqueezeOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(SqueezeOnnx);

tars::OpType SqueezeOnnx::opType() { return tars::OpType_Squeeze; }
tars::OpParameter SqueezeOnnx::type() { return tars::OpParameter_SqueezeParam; }

void SqueezeOnnx::run(tars::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      OnnxScope* scope) {
  auto para = new tars::SqueezeParamT;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "axes") {
      para->squeezeDims.resize(attributeProto.ints_size());
      for (int i = 0; i < para->squeezeDims.size(); ++i) {
        para->squeezeDims[i] = attributeProto.ints(i);
      }
    }
  }

  dstOp->main.value = para;
}

REGISTER_CONVERTER(SqueezeOnnx, Squeeze);

//
//  SpaceToDepthOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(SpaceToDepthOnnx);

tars::OpType SpaceToDepthOnnx::opType() { return tars::OpType_SpaceToDepth; }

tars::OpParameter SpaceToDepthOnnx::type() {
  return tars::OpParameter_DepthSpaceParam;
}

void SpaceToDepthOnnx::run(tars::OpT* dstOp, const onnx::NodeProto* onnxNode,
                           OnnxScope* scope) {
  auto spaceToDepthParam = new tars::DepthSpaceParamT;

  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "blocksize") {
      spaceToDepthParam->blockSize = (int)attributeProto.i();
    }
  }

  dstOp->main.value = spaceToDepthParam;
}

REGISTER_CONVERTER(SpaceToDepthOnnx, SpaceToDepth);

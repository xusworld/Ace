//
//  DepthToSpaceOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include <stdio.h>

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(DepthToSpaceOnnx);

tars::OpType DepthToSpaceOnnx::opType() { return tars::OpType_DepthToSpace; }

tars::OpParameter DepthToSpaceOnnx::type() {
  return tars::OpParameter_DepthSpaceParam;
}

void DepthToSpaceOnnx::run(tars::OpT* dstOp, const onnx::NodeProto* onnxNode,
                           OnnxScope* scope) {
  auto spaceToDepthParam = new tars::DepthSpaceParamT;

  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "blocksize") {
      spaceToDepthParam->blockSize = (int)attributeProto.i();
    } else if (attributeName == "mode") {
      std::map<const std::string, tars::DepthToSpaceMode> strToMode = {
          {"DCR", tars::DepthToSpaceMode_DCR},
          {"CRD", tars::DepthToSpaceMode_CRD}};
      const std::string& modeStr = attributeProto.s();
      if (strToMode.find(modeStr) != strToMode.end()) {
        spaceToDepthParam->mode = strToMode[modeStr];
      } else {
        MNN_ERROR("ONNX DepthToSpace mode [%s] is currently not supported.\n",
                  modeStr.c_str());
      }
    }
  }

  dstOp->main.value = spaceToDepthParam;
}

REGISTER_CONVERTER(DepthToSpaceOnnx, DepthToSpace);

//
//  ROIPoolingOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2021/10/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ROIPoolingOnnx);

tars::OpType ROIPoolingOnnx::opType() { return tars::OpType_ROIPooling; }

tars::OpParameter ROIPoolingOnnx::type() {
  return tars::OpParameter_RoiParameters;
}

void ROIPoolingOnnx::run(tars::OpT *dstOp, const onnx::NodeProto *onnxNode,
                         OnnxScope *scope) {
  auto roiPool = new tars::RoiParametersT;
  roiPool->spatialScale = 1;
  roiPool->poolType = tars::PoolType_MAXPOOL;

  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "pooled_shape") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS)
          << "Node Attribute ERROR";
      DCHECK(attributeProto.ints_size() == 2) << "Node Attribute ERROR";
      roiPool->pooledHeight = attributeProto.ints(0);
      roiPool->pooledWidth = attributeProto.ints(1);
    } else if (attributeName == "spatial_scale") {
      DCHECK(attributeProto.type() ==
             ::onnx::AttributeProto_AttributeType_FLOAT)
          << "Node Attribute ERROR";
      roiPool->spatialScale = attributeProto.f();
    } else {
      DLOG(ERROR) << "TODO!";
    }
  }

  dstOp->main.value = roiPool;
};

REGISTER_CONVERTER(ROIPoolingOnnx, MaxRoiPool);

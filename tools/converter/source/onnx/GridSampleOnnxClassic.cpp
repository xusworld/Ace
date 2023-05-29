//
//  GridSampleOnnxClassic.cpp
//  MNNConverter
//
//  Created by MNN on 2022/05/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(GridSampleOnnxClassic);

tars::OpType GridSampleOnnxClassic::opType() { return tars::OpType_GridSample; }

tars::OpParameter GridSampleOnnxClassic::type() {
  return tars::OpParameter_GridSample;
}

void GridSampleOnnxClassic::run(tars::OpT *dstOp,
                                const onnx::NodeProto *onnxNode,
                                OnnxScope *scope) {
  auto gridSampleParam = new tars::GridSampleT;

  gridSampleParam->mode = tars::SampleMode_BILINEAR;
  gridSampleParam->paddingMode = tars::BorderMode_ZEROS;
  gridSampleParam->alignCorners = false;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "mode") {
      gridSampleParam->mode = tars::SampleMode_BILINEAR;
      if (attributeProto.s() == "bilinear") {
        gridSampleParam->mode = tars::SampleMode_BILINEAR;
      } else if (attributeProto.s() == "nearest") {
        gridSampleParam->mode = tars::SampleMode_NEAREST;
      } else {
        LOG_INFO.stream() << "Don't support mode " << attributeProto.s();
      }
    }
    if (attributeName == "padding_mode") {
      gridSampleParam->paddingMode = tars::BorderMode_ZEROS;
      if (attributeProto.s() == "zeros") {
        gridSampleParam->paddingMode = tars::BorderMode_ZEROS;
      } else if (attributeProto.s() == "border") {
        gridSampleParam->paddingMode = tars::BorderMode_CLAMP;
      } else if (attributeProto.s() == "reflection") {
        gridSampleParam->paddingMode = tars::BorderMode_REFLECTION;
      } else {
        LOG_INFO.stream() << "Don't support padding_mode "
                          << attributeProto.s();
      }
    }
    if (attributeName == "align_corners") {
      gridSampleParam->alignCorners = attributeProto.i();
    }
  }

  dstOp->main.value = gridSampleParam;
}

REGISTER_CONVERTER(GridSampleOnnxClassic, GridSample);

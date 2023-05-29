//
//  GridSampleOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(GridSampleOnnx);

tars::OpType GridSampleOnnx::opType() { return tars::OpType_GridSample; }

tars::OpParameter GridSampleOnnx::type() {
  return tars::OpParameter_GridSample;
}

void GridSampleOnnx::run(tars::OpT *dstOp, const onnx::NodeProto *onnxNode,
                         OnnxScope *scope) {
  auto gridSampleParam = new tars::GridSampleT;

  gridSampleParam->mode = tars::SampleMode_BILINEAR;
  gridSampleParam->paddingMode = tars::BorderMode_ZEROS;
  gridSampleParam->alignCorners = false;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "mode") {
      switch (attributeProto.i()) {
        case 0:
          gridSampleParam->mode = tars::SampleMode_BILINEAR;
          break;
        case 1:
          gridSampleParam->mode = tars::SampleMode_NEAREST;
          break;
        default:
          LOG(FATAL) << "Unknown mode for " << onnxNode->name() << "!";
          break;
      }
    }
    if (attributeName == "padding_mode") {
      switch (attributeProto.i()) {
        case 0:
          gridSampleParam->paddingMode = tars::BorderMode_ZEROS;
          break;
        case 1:
          gridSampleParam->paddingMode = tars::BorderMode_CLAMP;
          break;
        case 2:
          gridSampleParam->paddingMode = tars::BorderMode_REFLECTION;
          break;
        default:
          LOG(FATAL) << "Unknown padding mode for " << onnxNode->name() << "!";
          break;
      }
    }
    if (attributeName == "align_corners") {
      gridSampleParam->alignCorners = attributeProto.i();
    }
  }

  dstOp->main.value = gridSampleParam;
}

// REGISTER_CONVERTER(GridSampleOnnx, GridSample);

// When we export torch.nn.functional.grid_sample to onnx, it's called
// GridSampler rather than GridSample, thus, we have to add the "r"
#define REGISTER_CONVERTER_r(name, opType) \
  static onnxOpConverterRegister<name> _Convert_##opType(#opType "r")
REGISTER_CONVERTER_r(GridSampleOnnx, GridSample);

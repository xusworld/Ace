//
//  ResizeBilinear.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "TfliteUtils.hpp"
#include "liteOpConverter.hpp"
using namespace tflite;
DECLARE_OP_COVERTER(ResizeBilinear);

tars::OpType ResizeBilinear::opType(bool quantizedModel) {
  DCHECK(!quantizedModel);
  if (quantizedModel) return tars::OpType_Interp;
  return tars::OpType_Interp;
}
tars::OpParameter ResizeBilinear::type(bool quantizedModel) {
  DCHECK(!quantizedModel);
  if (quantizedModel) return tars::OpParameter_Interp;
  return tars::OpParameter_Interp;
}

void ResizeBilinear::run(
    tars::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet,
    bool quantizedModel) {
  DCHECK(!quantizedModel);
  auto resizeParam = new tars::InterpT;
  const auto &scaleTensor = tfliteTensors[tfliteOp->inputs[1]];
  auto code = tfliteOpSet[tfliteOp->opcode_index]->builtin_code;
  if (BuiltinOperator_RESIZE_NEAREST_NEIGHBOR == code) {
    const auto &nearest =
        tfliteOp->builtin_options.AsResizeNearestNeighborOptions();
    resizeParam->resizeType = 1;
    resizeParam->alignCorners = nearest->align_corners;
    if (nearest->half_pixel_centers) {
      resizeParam->ctm = tars::CoordinateTransformationMode_HalfPixels;
    }
  } else if (BuiltinOperator_RESIZE_BILINEAR == code) {
    const auto &resizeOption =
        tfliteOp->builtin_options.AsResizeBilinearOptions();
    resizeParam->resizeType = 2;
    resizeParam->alignCorners = resizeOption->align_corners;
    if (resizeOption->half_pixel_centers) {
      resizeParam->ctm = tars::CoordinateTransformationMode_HalfPixels;
    }
  } else {
    DCHECK(false);
  }
  auto scaleDataPtr = reinterpret_cast<const int *>(
      tfliteModelBuffer[scaleTensor->buffer]->data.data());

  resizeParam->outputHeight = scaleDataPtr[0];
  resizeParam->outputWidth = scaleDataPtr[1];

  resizeParam->widthScale = 1.0;
  resizeParam->heightScale = 1.0;

  // set input output index
  dstOp->inputIndexes.resize(1);
  dstOp->outputIndexes.resize(1);
  dstOp->inputIndexes[0] = tfliteOp->inputs[0];
  dstOp->outputIndexes[0] = tfliteOp->outputs[0];

  dstOp->main.value = resizeParam;
}

using namespace tflite;
REGISTER_CONVERTER(ResizeBilinear, BuiltinOperator_RESIZE_BILINEAR);
REGISTER_CONVERTER(ResizeBilinear, BuiltinOperator_RESIZE_NEAREST_NEIGHBOR);

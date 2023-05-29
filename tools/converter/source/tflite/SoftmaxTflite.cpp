//
//  SoftmaxTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(SoftmaxTflite);

tars::OpType SoftmaxTflite::opType(bool quantizedModel) {
  if (quantizedModel) return tars::OpType_QuantizedSoftmax;
  return tars::OpType_Softmax;
}
tars::OpParameter SoftmaxTflite::type(bool quantizedModel) {
  if (quantizedModel) return tars::OpParameter_QuantizedSoftmax;
  return tars::OpParameter_Axis;
}

void SoftmaxTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  DCHECK(tfliteOp->inputs.size() == 1) << "Tflite Softmax input ERROR!";
  const auto& tfliteSoftmaxOption =
      tfliteOp->builtin_options.AsSoftmaxOptions();

  if (quantizedModel) {
    auto softmaxParamQuan = new tars::QuantizedSoftmaxT();
    softmaxParamQuan->beta = tfliteSoftmaxOption->beta;
    // input
    const int inputIndex = tfliteOp->inputs[0];
    const auto& inputTensor = tfliteTensors[inputIndex];
    softmaxParamQuan->inputScale = inputTensor->quantization->scale[0];
    dstOp->main.value = softmaxParamQuan;
  } else {
    auto paramFloat = new tars::AxisT;
    paramFloat->axis = -1;
    dstOp->main.value = paramFloat;
  }
  // set input output index
  dstOp->inputIndexes.resize(1);
  dstOp->outputIndexes.resize(1);
  dstOp->inputIndexes[0] = tfliteOp->inputs[0];
  dstOp->outputIndexes[0] = tfliteOp->outputs[0];
}

using namespace tflite;
REGISTER_CONVERTER(SoftmaxTflite, BuiltinOperator_SOFTMAX);

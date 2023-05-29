//
//  DequantizeTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2020/05/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfliteUtils.hpp"
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(DequantizeTflite);

tars::OpType DequantizeTflite::opType(bool quantizedModel) {
  return tars::OpType_Dequantize;
}

tars::OpParameter DequantizeTflite::type(bool quantizedModel) {
  return tars::OpParameter_Dequantize;
}

void DequantizeTflite::run(
    tars::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet,
    bool quantizedModel) {
  DCHECK(1 == tfliteOp->inputs.size())
      << "Dequantize should have one input now";
  auto inputIndex = tfliteOp->inputs[0];
  const auto &inputTensor = tfliteTensors[inputIndex];
  if (inputTensor->quantization.get() == nullptr ||
      inputTensor->quantization->zero_point.empty()) {
    // It's half to float / float to half, just use cast
    dstOp->type = tars::OpType_Cast;
    dstOp->main.type = tars::OpParameter_CastParam;
    dstOp->main.value = new tars::CastParamT;
    dstOp->main.AsCastParam()->srcT = tars::DataType_DT_FLOAT;
    dstOp->main.AsCastParam()->dstT = tars::DataType_DT_FLOAT;
    return;
  }

  auto dequantizeParam = new tars::DequantizeT;

  dequantizeParam->modelFormat = tars::ModeFormat_TFLITE;

  dequantizeParam->type = TfliteDequantDataTypeToMNN(inputTensor->type);

  auto quantizedParam = new tars::QuantizedParamT;

  quantizedParam->zeroPoint =
      static_cast<int32_t>(inputTensor->quantization->zero_point[0]);
  quantizedParam->scale = inputTensor->quantization->scale[0];
  dequantizeParam->inputQuantizedParam =
      std::unique_ptr<tars::QuantizedParamT>(quantizedParam);

  dstOp->main.value = dequantizeParam;
}

using namespace tflite;
REGISTER_CONVERTER(DequantizeTflite, BuiltinOperator_DEQUANTIZE);

//
//  PoolingTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "TfliteUtils.hpp"
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(PoolingTflite);

tars::OpType PoolingTflite::opType(bool quantizedModel) {
  if (quantizedModel) return tars::OpType_QuantizedAvgPool;
  return tars::OpType_Pooling;
}
tars::OpParameter PoolingTflite::type(bool quantizedModel) {
  if (quantizedModel) return tars::OpParameter_QuantizedAvgPool;
  return tars::OpParameter_Pool;
}

void PoolingTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  const auto& tflitePoolOption = tfliteOp->builtin_options.AsPool2DOptions();

  if (quantizedModel) {
    auto quantizedAvgPoolQuan = new tars::QuantizedAvgPoolT;
    quantizedAvgPoolQuan->modelFormat = tars::ModeFormat_TFLITE;

    quantizedAvgPoolQuan->kernelX = tflitePoolOption->filter_width;
    ;
    quantizedAvgPoolQuan->kernelY = tflitePoolOption->filter_height;

    quantizedAvgPoolQuan->strideX = tflitePoolOption->stride_w;
    quantizedAvgPoolQuan->strideY = tflitePoolOption->stride_h;

    // output
    const int outputIndex = tfliteOp->outputs[0];
    const auto& outputTensor = tfliteTensors[outputIndex];

    CalculateActivationRangeUint8(
        (tars::FusedActivation)tflitePoolOption->fused_activation_function,
        outputTensor->quantization, &quantizedAvgPoolQuan->outputActivationMin,
        &quantizedAvgPoolQuan->outputActivationMax);

    if (tflitePoolOption->padding == tflite::Padding_SAME) {
      quantizedAvgPoolQuan->padType = tars::PoolPadType_SAME;
    } else if (tflitePoolOption->padding == tflite::Padding_VALID) {
      quantizedAvgPoolQuan->padType = tars::PoolPadType_VALID;
    }
    dstOp->main.value = quantizedAvgPoolQuan;
  } else {
    DCHECK(tflitePoolOption->fused_activation_function ==
           tflite::ActivationFunctionType_NONE);
    auto poolParam = new tars::PoolT;
    poolParam->kernelX = tflitePoolOption->filter_width;
    poolParam->kernelY = tflitePoolOption->filter_height;
    poolParam->strideY = tflitePoolOption->stride_h;
    poolParam->strideX = tflitePoolOption->stride_w;
    if (tflitePoolOption->padding == tflite::Padding_SAME) {
      poolParam->padType = tars::PoolPadType_SAME;
    } else if (tflitePoolOption->padding == tflite::Padding_VALID) {
      poolParam->padType = tars::PoolPadType_VALID;
    }

    poolParam->type = tars::PoolType_AVEPOOL;
    const auto opIndex = tfliteOp->opcode_index;
    auto opType = tfliteOpSet[opIndex]->builtin_code;
    if (opType == tflite::BuiltinOperator_MAX_POOL_2D) {
      poolParam->type = tars::PoolType_MAXPOOL;
    }

    poolParam->isGlobal = false;
    dstOp->main.value = poolParam;
  }

  DCHECK(tfliteOp->inputs.size() == 1) << "Tflite pooling input ERROR";

  // set input output index
  dstOp->inputIndexes.resize(1);
  dstOp->outputIndexes.resize(1);
  dstOp->inputIndexes[0] = tfliteOp->inputs[0];
  dstOp->outputIndexes[0] = tfliteOp->outputs[0];
}

using namespace tflite;
REGISTER_CONVERTER(PoolingTflite, BuiltinOperator_AVERAGE_POOL_2D);
REGISTER_CONVERTER(PoolingTflite, BuiltinOperator_MAX_POOL_2D);

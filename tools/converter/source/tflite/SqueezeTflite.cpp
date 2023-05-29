//
//  SqueezeTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "TfliteUtils.hpp"
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(SqueezeTflite);

tars::OpType SqueezeTflite::opType(bool quantizedModel) {
  DCHECK(!quantizedModel);
  if (quantizedModel) return tars::OpType_Squeeze;
  return tars::OpType_Squeeze;
}
tars::OpParameter SqueezeTflite::type(bool quantizedModel) {
  DCHECK(!quantizedModel);
  if (quantizedModel) return tars::OpParameter_SqueezeParam;
  return tars::OpParameter_SqueezeParam;
}

void SqueezeTflite::run(
    tars::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet,
    bool quantizedModel) {
  DCHECK(!quantizedModel);
  auto squeezeParam = new tars::SqueezeParamT;
  const auto &squeezeOption = tfliteOp->builtin_options.AsSqueezeOptions();
  squeezeParam->squeezeDims = squeezeOption->squeeze_dims;

  // set input output index
  dstOp->inputIndexes.resize(1);
  dstOp->outputIndexes.resize(1);
  dstOp->inputIndexes[0] = tfliteOp->inputs[0];
  dstOp->outputIndexes[0] = tfliteOp->outputs[0];
  dstOp->main.value = squeezeParam;
}

using namespace tflite;
REGISTER_CONVERTER(SqueezeTflite, BuiltinOperator_SQUEEZE);

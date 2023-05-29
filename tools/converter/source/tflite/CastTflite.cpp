//
//  CastTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2020/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfliteUtils.hpp"
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(CastTflite);

tars::OpType CastTflite::opType(bool quantizedModel) {
  return tars::OpType_Cast;
}

tars::OpParameter CastTflite::type(bool quantizedModel) {
  return tars::OpParameter_CastParam;
}

void CastTflite::run(
    tars::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet,
    bool quantizedModel) {
  auto param = new tars::CastParamT;

  auto tfliteParam = tfliteOp->builtin_options.AsCastOptions();
  if (nullptr != tfliteParam) {
    param->srcT = TfliteDataTypeToMNN(tfliteParam->in_data_type);
    param->dstT = TfliteDataTypeToMNN(tfliteParam->out_data_type);
  } else {
    // Find type from tensor
    auto output = tfliteTensors[tfliteOp->outputs[0]].get();
    param->dstT = TfliteDataTypeToMNN(output->type);
    param->srcT = TfliteDataTypeToMNN(tfliteTensors[tfliteOp->inputs[0]]->type);
  }
  dstOp->main.value = param;
}

using namespace tflite;
REGISTER_CONVERTER(CastTflite, BuiltinOperator_CAST);

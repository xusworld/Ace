//
//  UnaryTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(UnaryTflite);
tars::OpType UnaryTflite::opType(bool quantizedModel) {
  return tars::OpType_UnaryOp;
}
tars::OpParameter UnaryTflite::type(bool quantizedModel) {
  return tars::OpParameter_UnaryOp;
}

static tars::UnaryOpOperation _convert(tflite::BuiltinOperator op) {
#define MNNCONVERT(x, y) \
  if (op == tflite::BuiltinOperator_##x) return tars::UnaryOpOperation_##y;
  MNNCONVERT(ABS, ABS);
  MNNCONVERT(COS, COS);
  MNNCONVERT(CEIL, CEIL);
  MNNCONVERT(EXP, EXP);
  MNNCONVERT(FLOOR, FLOOR);
  MNNCONVERT(HARD_SWISH, HARDSWISH);
  MNNCONVERT(LOG, LOG);
  MNNCONVERT(NEG, NEG);
  MNNCONVERT(ROUND, ROUND);
  MNNCONVERT(RSQRT, RSQRT);
  MNNCONVERT(SQUARE, SQUARE);
  MNNCONVERT(SQRT, SQRT);
  MNNCONVERT(SIN, SIN);
#undef MNNCONVERT
  return (tars::UnaryOpOperation)0;
}
void UnaryTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  auto param = new tars::UnaryOpT;
  param->opType = _convert(tfliteOpSet[tfliteOp->opcode_index]->builtin_code);
  dstOp->main.value = param;
}

using namespace tflite;

#define MNNCONVERT(x, y) REGISTER_CONVERTER(UnaryTflite, BuiltinOperator_##x);
MNNCONVERT(ABS, ABS);
MNNCONVERT(COS, COS);
MNNCONVERT(CEIL, CEIL);
MNNCONVERT(EXP, EXP);
MNNCONVERT(FLOOR, FLOOR);
MNNCONVERT(HARD_SWISH, HARDSWISH);
MNNCONVERT(LOG, LOG);
MNNCONVERT(NEG, NEG);
MNNCONVERT(ROUND, ROUND);
MNNCONVERT(RSQRT, RSQRT);
MNNCONVERT(SQUARE, SQUARE);
MNNCONVERT(SQRT, SQRT);
MNNCONVERT(SIN, SIN);
#undef MNNCONVERT

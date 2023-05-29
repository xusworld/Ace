//
//  SpaceToDepthTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "liteOpConverter.hpp"
using namespace tflite;

DECLARE_OP_COVERTER(SpaceToDepthTflite);

tars::OpType SpaceToDepthTflite::opType(bool quantizedModel) {
  return tars::OpType_SpaceToDepth;
}
tars::OpParameter SpaceToDepthTflite::type(bool quantizedModel) {
  return tars::OpParameter_DepthSpaceParam;
}

void SpaceToDepthTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  auto spaceToDepthParam = new tars::DepthSpaceParamT;
  auto opt = tfliteOp->builtin_options.AsSpaceToDepthOptions();
  spaceToDepthParam->blockSize = opt->block_size;
  dstOp->main.value = spaceToDepthParam;
}

REGISTER_CONVERTER(SpaceToDepthTflite, BuiltinOperator_SPACE_TO_DEPTH);

/**
 From https://github.com/alibaba/MNN/issues/1452
 Author: https://github.com/pkjq11
 */

DECLARE_OP_COVERTER(DepthToSpaceTflite);

tars::OpType DepthToSpaceTflite::opType(bool quantizedModel) {
  return tars::OpType_DepthToSpace;
}
tars::OpParameter DepthToSpaceTflite::type(bool quantizedModel) {
  return tars::OpParameter_DepthSpaceParam;
}

void DepthToSpaceTflite::run(
    tars::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
    bool quantizedModel) {
  auto depthToSpaceParam = new tars::DepthSpaceParamT;
  auto opt = tfliteOp->builtin_options.AsDepthToSpaceOptions();
  depthToSpaceParam->blockSize = opt->block_size;
  dstOp->main.value = depthToSpaceParam;
}

REGISTER_CONVERTER(DepthToSpaceTflite, BuiltinOperator_DEPTH_TO_SPACE);

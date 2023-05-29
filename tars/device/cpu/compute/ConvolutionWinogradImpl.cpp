//
//  ConvolutionWinogradImpl.cpp
//  MNN
//
//  Created by MNN on 2022/01/20.
//  Copyright © 2018 - 2022, Alibaba Group Holding Limited
//

#include <math.h>

#include <MNN/AutoTime.hpp>

#include "common/MemoryFormater.h"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "device/cpu/compute/CommonOptFunction.h"
#include "device/cpu/compute/ConvOpt.h"
#include "device/cpu/compute/ConvolutionWinogradImpl.hpp"
#include "math/WingoradGenerater.hpp"

// #define MNN_WINOGRAD_PRINT_REDUCE_RATE
// #define MNN_WINO_TRANFORM_TEST_CLOSE
namespace tars {
ConvolutionWinogradImpl::ConvolutionWinogradImpl(
    const Convolution2DCommon *convOp, Device *b)
    : tars::CPUConvolution(convOp, b) {}

ConvolutionWinogradImpl::~ConvolutionWinogradImpl() {}

WinogradConfig ConvolutionWinogradImpl::bestWinogradUnit(
    const Convolution2DCommon *common, const Tensor *inputTensor,
    const Tensor *outputTensor, int threadNumber, Device *b,
    const PerfConfig &denseConfig) {
  return WinogradConfig();
}

bool ConvolutionWinogradImpl::canUseWinograd(
    const Convolution2DCommon *common) {
  if (common->kernelY() != common->kernelX() || common->kernelY() <= 1) {
    return false;
  }
  if (common->dilateX() != 1 || common->dilateY() != 1) {
    return false;
  }
  if (common->strideX() != 1 || common->strideY() != 1) {
    return false;
  }
  return true;
}

Status ConvolutionWinogradImpl::onExecute(
    const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
  return Status::OK();
}

Status ConvolutionWinogradImpl::onResize(const std::vector<Tensor *> &inputs,
                                         const std::vector<Tensor *> &outputs) {
  return Status::OK();
}

bool ConvolutionWinogradImpl::onClone(Device *bn, const Op *op,
                                      Operation **dst) {
  return false;
}

}  // namespace tars

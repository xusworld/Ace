//
//  ConvolutionWinogradBridge.cpp
//  MNN
//
//  Created by MNN on 2022/01/20.
//  Copyright © 2018 - 2022, Alibaba Group Holding Limited
//

#include "device/cpu/CPUConvolution.hpp"
#include "device/cpu/compute/ConvolutionPackFreeWinograd.hpp"
#include "device/cpu/compute/ConvolutionPackWinograd.hpp"
#include "device/cpu/compute/ConvolutionWinogradBridge.hpp"
#include "device/cpu/compute/ConvolutionWinogradImpl.hpp"

namespace tars {

WinogradConfig ConvolutionWinogradBridge::bestWinogradUnit(
    const Convolution2DCommon *common, const Tensor *inputTensor,
    const Tensor *outputTensor, int threadNumber, Device *b,
    const PerfConfig &denseConfig) {
//  Currently packfree is only used in x86 architecture
#ifdef MNN_USE_SSE
  auto core = static_cast<CPUDevice *>(b)->functions();
  if (16 == core->pack) {  // avx512
    return ConvolutionPackFreeWinograd::bestWinogradUnit(
        common, inputTensor, outputTensor, threadNumber, b, denseConfig);
  } else {
#endif
    return ConvolutionPackWinograd::bestWinogradUnit(
        common, inputTensor, outputTensor, threadNumber, b, denseConfig);

#ifdef MNN_USE_SSE
  }
#endif
}

bool ConvolutionWinogradBridge::canUseWinograd(
    const Convolution2DCommon *common) {
  return ConvolutionPackWinograd::canUseWinograd(common);
}

ConvolutionWinogradImpl *ConvolutionWinogradBridge::createWinogradImpl(
    const Convolution2DCommon *common, const Tensor *input,
    const Tensor *output, Device *b, const float *originWeight,
    size_t originWeightSize, const float *bias, size_t biasSize,
    WinogradConfig config) {
#ifdef MNN_USE_SSE
  auto core = static_cast<CPUDevice *>(b)->functions();
  // Adopt different algorithm for x86 and arm
  if (16 == core->pack) {  // avx512
    return new ConvolutionPackFreeWinograd(common, input, output, b,
                                           originWeight, originWeightSize, bias,
                                           biasSize, config);
  } else {
#endif

    return new ConvolutionPackWinograd(common, input, output, b, originWeight,
                                       originWeightSize, bias, biasSize,
                                       config);
#ifdef MNN_USE_SSE
  }
#endif
}

}  // namespace tars

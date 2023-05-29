//
//  ConvolutionWinogradBridge
//  MNN
//
//  Created by MNN on 2022/01/20.
//  Copyright © 2018 - 2022, Alibaba Group Holding Limited
//

#ifndef ConvolutionWinogradBridge_hpp
#define ConvolutionWinogradBridge_hpp

#include "device/cpu/CPUConvolution.hpp"
#include "device/cpu/compute/ConvolutionWinogradImpl.hpp"

namespace tars {

class ConvolutionWinogradBridge {
 public:
  static bool canUseWinograd(const Convolution2DCommon *convOp);

  static WinogradConfig bestWinogradUnit(const Convolution2DCommon *convOp,
                                         const Tensor *input,
                                         const Tensor *output, int threadnumber,
                                         Device *b,
                                         const PerfConfig &denseConfig);

  static ConvolutionWinogradImpl *createWinogradImpl(
      const Convolution2DCommon *convOp, const Tensor *input,
      const Tensor *output, Device *b, const float *originWeight,
      size_t originWeightSize, const float *bias, size_t biasSize,
      WinogradConfig config);
};

}  // namespace tars
#endif /* ConvolutionWinogradBridge_hpp */

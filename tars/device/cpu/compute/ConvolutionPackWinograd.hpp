//
//  ConvolutionPackWinograd.hpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionPackWinograd_hpp
#define ConvolutionPackWinograd_hpp

#include "device/cpu/CPUConvolution.hpp"
#include "device/cpu/compute/CommonOptFunction.h"
#include "device/cpu/compute/ConvolutionWinogradImpl.hpp"

#define CONVOLUTION_WINOGRAD_MAX_UNIT 8
#define CONVOLUTION_WINOGRAD_MIN_UNIT 2

namespace tars {
class ConvolutionPackWinograd : public ConvolutionWinogradImpl {
 public:
  ConvolutionPackWinograd(const Convolution2DCommon *convOp,
                          const Tensor *input, const Tensor *output, Device *b,
                          const float *originWeight, size_t originWeightSize,
                          const float *bias, size_t biasSize,
                          WinogradConfig config);
  virtual ~ConvolutionPackWinograd();
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;

  static WinogradConfig bestWinogradUnit(const Convolution2DCommon *convOp,
                                         const Tensor *input,
                                         const Tensor *output, int threadnumber,
                                         Device *b,
                                         const PerfConfig &denseConfig);
  virtual bool onClone(Device *bn, const Op *op, Operation **dst) override;

 private:
  ConvolutionPackWinograd(std::shared_ptr<CPUConvolution::Resource> resource,
                          const Convolution2DCommon *convOp, Device *b)
      : ConvolutionWinogradImpl(convOp, b) {
    mResource = resource;
  }
  std::pair<int, std::function<void(int tId, const uint8_t *, uint8_t *)>>
      mMainFunction;
  std::pair<int, std::function<void(int, uint8_t *)>> mPostFunction;
};
}  // namespace tars
#endif /* ConvolutionPackWinograd_hpp */

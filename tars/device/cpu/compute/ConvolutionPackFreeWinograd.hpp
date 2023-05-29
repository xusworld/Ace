//
//  ConvolutionPackFreeWinograd.hpp
//  MNN
//
//  Created by MNN on 2022/01/20.
//  Copyright © 2018 - 2022, Alibaba Group Holding Limited
//

#ifndef ConvolutionPackFreeWinograd_hpp
#define ConvolutionPackFreeWinograd_hpp

#include "device/cpu/CPUConvolution.hpp"
#include "device/cpu/compute/CommonOptFunction.h"
#include "device/cpu/compute/ConvolutionPackFreeWinograd.hpp"
#include "device/cpu/compute/ConvolutionWinogradImpl.hpp"

#define CONVOLUTION_WINOGRAD_MAX_UNIT 8
#define CONVOLUTION_WINOGRAD_MIN_UNIT 2

namespace tars {

class ConvolutionPackFreeWinograd : public ConvolutionWinogradImpl {
 public:
  ConvolutionPackFreeWinograd(const Convolution2DCommon *convOp,
                              const Tensor *input, const Tensor *output,
                              Device *b, const float *originWeight,
                              size_t originWeightSize, const float *bias,
                              size_t biasSize, WinogradConfig config);
  virtual ~ConvolutionPackFreeWinograd();
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  bool updateWinogradBuffer(const Tensor *input, const Tensor *output);
  static WinogradConfig bestWinogradUnit(const Convolution2DCommon *convOp,
                                         const Tensor *input,
                                         const Tensor *output, int threadnumber,
                                         Device *b,
                                         const PerfConfig &denseConfig);
  static WinogradConfig updateBestWinogradUnit(
      const Convolution2DCommon *convOp, const Tensor *input,
      const Tensor *output, int threadnumber, Device *b);
  virtual bool onClone(Device *bn, const Op *op, Operation **dst) override;

 private:
  ConvolutionPackFreeWinograd(
      std::shared_ptr<CPUConvolution::Resource> resource,
      const Convolution2DCommon *convOp, Device *b)
      : ConvolutionWinogradImpl(convOp, b) {
    mResource = resource;
  }
};
}  // namespace tars
#endif /* ConvolutionWinogradImpl_hpp */

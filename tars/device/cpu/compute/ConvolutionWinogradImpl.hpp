//
//  ConvolutionWinogradImpl.hpp
//  MNN
//
//  Created by MNN on 2022/01/20.
//  Copyright © 2018 - 2022, Alibaba Group Holding Limited
//

#ifndef ConvolutionWinogradImpl_hpp
#define ConvolutionWinogradImpl_hpp

#include "device/cpu/CPUConvolution.hpp"
#include "device/cpu/compute/CommonOptFunction.h"
#include "device/cpu/compute/ConvolutionFloatFactory.h"

#define CONVOLUTION_WINOGRAD_MAX_UNIT 8
#define CONVOLUTION_WINOGRAD_MIN_UNIT 2

namespace tars {

class WinogradConfig : public PerfConfig {
 public:
  WinogradConfig() : PerfConfig(), unit{0} {}
  WinogradConfig(int unit_, bool isParallelInner_, int eTile_, int ePack_,
                 int hPack_, float instructionCosts_)
      : PerfConfig(unit_, eTile_, ePack_, hPack_, instructionCosts_),
        unit{unit_} {}
  bool operator!=(const WinogradConfig &other) {
    return unit != other.unit || isParallelInner != other.isParallelInner ||
           ePack != other.ePack || eTile != other.eTile || hPack != other.hPack;
  }
  WinogradConfig &operator=(const WinogradConfig &other) {
    isParallelInner = other.isParallelInner;
    eTile = other.eTile;
    ePack = other.ePack;
    hPack = other.hPack;
    instructionCosts = other.instructionCosts;
    unit = other.unit;
    return *this;
  }
  int unit;  // output block size of winograd
};

class ConvolutionWinogradImpl : public CPUConvolution {
 public:
  ConvolutionWinogradImpl(const Convolution2DCommon *convOp, Device *b);
  virtual ~ConvolutionWinogradImpl();
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  static bool canUseWinograd(const Convolution2DCommon *convOp);
  static WinogradConfig bestWinogradUnit(const Convolution2DCommon *convOp,
                                         const Tensor *input,
                                         const Tensor *output, int threadnumber,
                                         Device *b,
                                         const PerfConfig &denseConfig);
  virtual bool onClone(Device *bn, const Op *op, Operation **dst) override;

 protected:
  ConvolutionWinogradImpl(std::shared_ptr<CPUConvolution::Resource> resource,
                          const Convolution2DCommon *convOp, Device *b)
      : CPUConvolution(convOp, b) {
    mResource = resource;
  }
  std::shared_ptr<CPUConvolution::Resource> mResource;
  std::shared_ptr<Tensor> mA;
  std::shared_ptr<Tensor> mB;

  std::shared_ptr<Tensor> mTempBuffer;
  std::shared_ptr<Tensor> mTransformMidBuffer;
  std::shared_ptr<Tensor> mGemmMidBuffer;

  CoreFunctions::WinoTransPackFunc mSourceTransformPack;
  CoreFunctions::WinoUnrollTransFunc mSourceUnrollTransform;
  std::shared_ptr<CoreFunctions::WinoUnrollDestTransFunc> mDestUnrollTransform;
  std::vector<float> mPostParameters;
  WinogradConfig mConvPerfconfig;
  const float *mOriginWeight =
      nullptr;  // only used for source transform when resize called. would be
                // invalid after release model.
};
}  // namespace tars
#endif /* ConvolutionWinogradImpl_hpp */

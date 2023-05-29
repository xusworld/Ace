//
//  ConvInt8Winograd.hpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvInt8Winograd_hpp
#define ConvInt8Winograd_hpp

#include "device/cpu/CPUConvolution.hpp"
#include "device/cpu/compute/Int8FunctionsOpt.h"

namespace tars {
class ConvInt8Winograd : public CPUConvolution {
  class WinoExecution;

 public:
  struct Unit {
    int kyStart;
    int kxStart;
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> output;
    std::shared_ptr<WinoExecution> runner;
  };
  ConvInt8Winograd(Device *b, const Convolution2D *convOp,
                   std::shared_ptr<ResourceInt8> res);
  virtual ~ConvInt8Winograd();
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;

  static bool mustUse(const Convolution2D *convOp);
  virtual bool onClone(Device *bn, const Op *op, Operation **dst) override;

 private:
  ConvInt8Winograd(Device *backend, const Convolution2DCommon *common,
                   const ConvInt8Winograd &exe);
  // subExecutions
  std::vector<Unit> mUnits;
  std::shared_ptr<CPUConvolution::ResourceInt8> mResource;
  std::shared_ptr<Tensor> mInputFloat;

  struct WinoResource {
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> offsets;
    std::shared_ptr<Tensor> scales;            // alpha2 * ROUND_UP(oc, UNIT)
    std::shared_ptr<Tensor> transInputScales;  // alpha2
    std::vector<int> transInputZeroPoints;
    Device *backend;
    ~WinoResource() {
      backend->onReleaseBuffer(weight.get(), Device::STATIC);
      backend->onReleaseBuffer(offsets.get(), Device::STATIC);
      backend->onReleaseBuffer(scales.get(), Device::STATIC);
      backend->onReleaseBuffer(transInputScales.get(), Device::STATIC);
    }
  };
  static std::shared_ptr<WinoResource> makeWinoResource(
      const int8_t *originWeight, std::shared_ptr<Tensor> scaleFloat,
      const int32_t *attr, Device *backend, int oc, int ic, int kernelY,
      int kernelX);
  class WinoExecution : public Operation {
   public:
    WinoExecution(std::shared_ptr<WinoResource> res, int kernelY, int kernelX,
                  int unitY, int unitX, int outputCount, int inputCount);

    WinoExecution(Device *bn, const WinoExecution &exe);
    virtual ~WinoExecution() = default;
    virtual Status onExecute(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;
    virtual Status onResize(const std::vector<Tensor *> &inputs,
                            const std::vector<Tensor *> &outputs) override;
    // weight
    std::shared_ptr<WinoResource> mWinoResource;
    // buffer
    std::shared_ptr<Tensor> mTempInputBuffer;
    std::shared_ptr<Tensor> mTempOutputBuffer;
    std::shared_ptr<Tensor> mTransformMidBuffer;
    int mUnitY, mUnitX;
    int mKernelY, mKernelX;
    int mPadY, mPadX;
    friend class ConvInt8Winograd;
  };
};
}  // namespace tars
#endif /* ConvInt8Winograd_hpp */

//
//  CPUConvolutionDepthwise.hpp
//  MNN
//
//  Created by MNN on 2018/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUConvolutionDepthwise_hpp
#define CPUConvolutionDepthwise_hpp

#include "core/AutoStorage.h"
#include "device/cpu/CPUConvolution.hpp"
#include "device/cpu/compute/ConvolutionIntFactory.hpp"

namespace tars {
class CPUConvolutionDepthwise {
 public:
  class BasicFloatExecution : public CPUConvolution {
   public:
    BasicFloatExecution(const Convolution2DCommon *common, Device *b)
        : CPUConvolution(common, b) {}
    virtual ~BasicFloatExecution() = default;
    virtual Status onExecute(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;
    virtual Status onResize(const std::vector<Tensor *> &inputs,
                            const std::vector<Tensor *> &outputs) override;

   private:
    std::function<void(const uint8_t *, uint8_t *, int)> mExecutor;
    int mNumber = 1;
  };
  class MultiInputFloatExecution : public BasicFloatExecution {
   public:
    MultiInputFloatExecution(const Convolution2DCommon *common, Device *b)
        : BasicFloatExecution(common, b) {}
    virtual ~MultiInputFloatExecution() = default;
    virtual Status onResize(const std::vector<Tensor *> &inputs,
                            const std::vector<Tensor *> &outputs) override;
    virtual Status onExecute(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;

   private:
    std::unique_ptr<Tensor> mWeight;
    std::unique_ptr<Tensor> mBias;
    std::vector<Tensor *> mTempInputs;
  };
  class FloatExecution : public CPUConvolution {
   public:
    FloatExecution(const Convolution2DCommon *common, Device *b,
                   const float *originWeight, size_t originWeightSize,
                   const float *bias, size_t biasSize);
    virtual ~FloatExecution();
    virtual Status onExecute(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override {
      return mOrigin->onExecute(mTempInputs, outputs);
    }
    virtual Status onResize(const std::vector<Tensor *> &inputs,
                            const std::vector<Tensor *> &outputs) override {
      mTempInputs = {inputs[0], mResource->mWeight.get(),
                     mResource->mBias.get()};
      return mOrigin->onResize(mTempInputs, outputs);
    }
    virtual bool onClone(Device *bn, const Op *op, Operation **dst) override;

   private:
    FloatExecution(std::shared_ptr<Resource> resource,
                   const Convolution2DCommon *common, Device *b)
        : CPUConvolution(common, b) {
      mResource = resource;
      mOrigin.reset(new BasicFloatExecution(common, b));
    }
    std::shared_ptr<CPUConvolution::Resource> mResource;
    std::vector<Tensor *> mTempInputs;
    std::unique_ptr<BasicFloatExecution> mOrigin;
  };
};
}  // namespace tars

#endif /* CPUConvolutionDepthwise_hpp */

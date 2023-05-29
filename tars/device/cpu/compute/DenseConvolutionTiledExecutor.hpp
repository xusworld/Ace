//
//  DenseConvolutionTiledExecutor
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DenseConvolutionTiledExecutor_hpp
#define DenseConvolutionTiledExecutor_hpp

#include <functional>

#include "ConvolutionTiledExecutor.hpp"
#include "device/cpu/CPUConvolution.hpp"
// Tiled Slide Window or Im2Col + GEMM
namespace tars {
class DenseConvolutionTiledImpl : public ConvolutionTiledImpl {
 public:
  DenseConvolutionTiledImpl(const Convolution2DCommon *common, Device *b)
      : ConvolutionTiledImpl(common, b) {
    // Do nothing
  }
  Status onResize(const std::vector<Tensor *> &inputs,
                  const std::vector<Tensor *> &outputs) override;
  Status onExecute(const std::vector<Tensor *> &inputs,
                   const std::vector<Tensor *> &outputs) override;
  virtual ~DenseConvolutionTiledImpl() = default;
  void getPackParameter(int *eP, int *lP, int *hP,
                        const CoreFunctions *core) override;
  static PerfConfig bestTileConvolutionConfig(const Convolution2DCommon *common,
                                              const Tensor *inputTensor,
                                              const Tensor *outputTensor,
                                              int threadNumber, Device *b);

 protected:
};
class DenseConvolutionTiledExecutor : public ConvolutionTiledExecutor {
 public:
  DenseConvolutionTiledExecutor(const Convolution2DCommon *common, Device *b,
                                const float *originWeight,
                                size_t originWeightSize, const float *bias,
                                size_t biasSize);

  DenseConvolutionTiledExecutor(std::shared_ptr<CPUConvolution::Resource> res,
                                const Convolution2DCommon *common, Device *b);
  virtual ~DenseConvolutionTiledExecutor();

  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override {
    return mProxy->onExecute(inputs, outputs);
  }
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override {
    mInputs = {inputs[0], mResource->mWeight.get(), mResource->mBias.get()};
    return mProxy->onResize(mInputs, outputs);
  }
  virtual bool onClone(Device *bn, const Op *op, Operation **dst) override;
  void initWeight(float *dest, const float *source, float *cache, int depth,
                  int outputCount, int kernelSize,
                  const CoreFunctions *function);
  static PerfConfig bestTileConvolutionConfig(const Convolution2DCommon *common,
                                              const Tensor *inputTensor,
                                              const Tensor *outputTensor,
                                              int threadNumber, Device *b) {
    return DenseConvolutionTiledImpl::bestTileConvolutionConfig(
        common, inputTensor, outputTensor, threadNumber, b);
  }

 protected:
  std::shared_ptr<DenseConvolutionTiledImpl> mProxy;
};

class ConvolutionTiledExecutorMultiInput : public Operation {
 public:
  ConvolutionTiledExecutorMultiInput(const Convolution2DCommon *common,
                                     Device *b)
      : Operation(b) {
    mProxy.reset(new DenseConvolutionTiledImpl(common, b));
  }
  virtual ~ConvolutionTiledExecutorMultiInput() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;

 private:
  std::shared_ptr<Tensor> mTempWeight;
  std::shared_ptr<Tensor> mTempWeightCache;
  std::shared_ptr<Tensor> mTempBias;
  std::shared_ptr<DenseConvolutionTiledImpl> mProxy;
  std::vector<Tensor *> mInputs;
};

}  // namespace tars

#endif /* DenseConvolutionTiledExecutor_hpp */

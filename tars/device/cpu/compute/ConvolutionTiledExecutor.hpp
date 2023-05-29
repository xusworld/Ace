//
//  ConvolutionTiledExecutor.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionTiledExecutor_hpp
#define ConvolutionTiledExecutor_hpp

#include <functional>

#include "device/cpu/CPUConvolution.hpp"
// Tiled Slide Window or Im2Col + GEMM
namespace tars {
class ConvolutionTiledImpl : public CPUConvolution {
 public:
  ConvolutionTiledImpl(const Convolution2DCommon *common, Device *b)
      : CPUConvolution(common, b) {
    // Do nothing
  }
  virtual ~ConvolutionTiledImpl() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual void getPackParameter(int *eP, int *lP, int *hP,
                                const CoreFunctions *core) = 0;

 protected:
  Tensor mTempBufferTranspose;
  std::pair<int, std::function<void(int)>> mFunction;
};

class ConvolutionTiledExecutor : public Operation {
 public:
  ConvolutionTiledExecutor(Device *b, const float *bias, size_t biasSize);
  ConvolutionTiledExecutor(std::shared_ptr<CPUConvolution::Resource> res,
                           Device *b);
  virtual ~ConvolutionTiledExecutor();

  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override {
    return Status::ERROR();
  }
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override {
    return Status::ERROR();
  }
  virtual bool onClone(Device *bn, const Op *op, Operation **dst) override;
  void initWeight(const float *source, float *cache, int depth, int outputCount,
                  int kernelSize, const CoreFunctions *function);

 protected:
  std::vector<Tensor *> mInputs;
  std::shared_ptr<CPUConvolution::Resource> mResource;
};

}  // namespace tars

#endif /* ConvolutionTiledExecutor_hpp */

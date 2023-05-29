//
//  ConvolutionInt8Executor.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionInt8Executor_hpp
#define ConvolutionInt8Executor_hpp

#include <stdio.h>

#include "core/AutoStorage.h"
#include "device/cpu/CPUConvolution.hpp"
#include "device/cpu/compute/ConvolutionFloatFactory.h"
#include "device/cpu/compute/ConvolutionIntFactory.hpp"

namespace tars {
class ConvolutionInt8Executor : public CPUConvolution {
 public:
  ConvolutionInt8Executor(const Convolution2DCommon *convOp, Device *b,
                          const ConvolutionCommon::Int8Common *common,
                          const float *bias, size_t biasSize);
  virtual ~ConvolutionInt8Executor();
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;

 private:
  std::shared_ptr<Tensor> mWeight;
  AutoStorage<float> mAlpha;
  AutoStorage<float> mBias;
  const IDSTQuan *mQuan;
  Tensor mSrcCopyBuffer;

  Tensor mTempBuffer;
  Tensor mTempDstBuffer;
  ConvolutionCommon::Im2ColParameter mIm2ColParamter;
  int mSrcCount;
  float mAMin;
  float mAMax;
  float mQuanScale;
  std::vector<float> mPostParameters;
  // mFakeBias used by GemmKernel
  std::shared_ptr<Tensor> mFakeBias;
};
}  // namespace tars

#endif /* ConvolutionInt8Executor_hpp */

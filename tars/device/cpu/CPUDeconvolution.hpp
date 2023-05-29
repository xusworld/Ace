//
//  CPUDeconvolution.hpp
//  MNN
//
//  Created by MNN on 2018/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDeconvolution_hpp
#define CPUDeconvolution_hpp

#include "CPUConvolution.hpp"
#include "compute/StrassenMatmulComputor.hpp"

namespace tars {
class CPUDeconvolutionBasic : public CPUConvolution {
 public:
  CPUDeconvolutionBasic(const Tensor *input, const Op *convOp, Device *b);
  virtual ~CPUDeconvolutionBasic() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;

 protected:
  int mSrcCount;
  std::vector<float> mPostParameters;
};

class CPUDeconvolutionCommon : public CPUDeconvolutionBasic {
 public:
  CPUDeconvolutionCommon(const Tensor *input, const Op *convOp, Device *b);
  virtual ~CPUDeconvolutionCommon();

 protected:
  std::shared_ptr<Tensor> mBias;
};

class CPUDeconvolutionOrigin : public CPUDeconvolutionBasic {
 public:
  CPUDeconvolutionOrigin(const Tensor *input, const Op *convOp, Device *b)
      : CPUDeconvolutionBasic(input, convOp, b) {
    // Do nothing
  }
  virtual ~CPUDeconvolutionOrigin() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;

 private:
  std::shared_ptr<StrassenMatrixComputor> mMatMul;
  std::vector<std::pair<std::function<void(float *, int)>, int>> mPostFunctions;
};

class CPUDeconvolution : public CPUDeconvolutionCommon {
 public:
  CPUDeconvolution(const Tensor *input, const Op *convOp, Device *b);
  virtual ~CPUDeconvolution();
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override {
    mOrigin->onExecute(mTempInputs, outputs);
    return Status::OK();
  }
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override {
    mTempInputs = {inputs[0], mWeight.get(), mBias.get()};
    return mOrigin->onResize(mTempInputs, outputs);
  }

 private:
  std::shared_ptr<Tensor> mWeight;
  std::vector<Tensor *> mTempInputs;
  std::shared_ptr<CPUDeconvolutionOrigin> mOrigin;
};
}  // namespace tars
#endif /* CPUDeconvolution_hpp */

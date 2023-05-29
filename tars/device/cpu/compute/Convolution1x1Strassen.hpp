//
//  Convolution1x1Strassen.hpp
//  MNN
//
//  Created by MNN on 2019/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Convolution1x1Strassen_hpp
#define Convolution1x1Strassen_hpp

#include <functional>

#include "device/cpu/CPUConvolution.hpp"
#include "device/cpu/compute/StrassenMatmulComputor.hpp"
namespace tars {
class Convolution1x1Strassen : public CPUConvolution {
 public:
  Convolution1x1Strassen(const Convolution2DCommon *common, Device *b,
                         const float *originWeight, size_t originWeightSize,
                         const float *bias, size_t biasSize);
  Convolution1x1Strassen(std::shared_ptr<CPUConvolution::Resource> resource,
                         const Convolution2DCommon *common, Device *b);
  virtual ~Convolution1x1Strassen();

  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual bool onClone(Device *bn, const Op *op, Operation **dst) override;

 private:
  std::shared_ptr<CPUConvolution::Resource> mResource;

  struct Unit {
    bool mValid = true;
    int offset[4];  // Input, Weight, Output, Bias
    std::shared_ptr<StrassenMatrixComputor> mStracssenComputor;
  };

  std::vector<Unit> mUnits;
};
}  // namespace tars

#endif /* Convolution1x1Strassen_hpp */

//
//  ConvolutionGroup.hpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionGroupInt8_hpp
#define ConvolutionGroupInt8_hpp

#include "device/cpu/compute/ConvolutionIntFactory.hpp"

namespace tars {
class ConvolutionGroup : public Operation {
 public:
  ConvolutionGroup(
      Device *b, const std::vector<std::shared_ptr<Operation>> &subConvolution);
  virtual ~ConvolutionGroup() = default;

  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;

  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  std::unique_ptr<Tensor> mInputRaw;
  std::unique_ptr<Tensor> mOutputRaw;

  std::unique_ptr<Tensor> mInputUnit;
  std::unique_ptr<Tensor> mOutputUnit;

  std::vector<Tensor *> mInputUnitWrap;
  std::vector<Tensor *> mOutputUnitWrap;
  std::vector<std::shared_ptr<Operation>> mSubConvolution;
};
}  // namespace tars

#endif /* ConvolutionGroupInt8_hpp */

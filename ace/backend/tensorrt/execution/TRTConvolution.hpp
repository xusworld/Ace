//
//  TRTConvolution.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTConvolution_HPP
#define MNN_TRTConvolution_HPP

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace ace {

class TRTConvolution : public TRTCommonExecution {
 public:
  TRTConvolution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                 const std::vector<Tensor *> &outputs);
  virtual ~TRTConvolution() = default;
  virtual std::vector<ITensor *> onEncode(
      const std::vector<ITensor *> &inputs) override;

 private:
  IActivationLayer *mActivationLayer{nullptr};
};

}  // namespace ace

#endif  // MNN_TRTConvolution_HPP

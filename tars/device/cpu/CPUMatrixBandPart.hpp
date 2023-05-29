//
//  CPUMatrixBandPart.hpp
//  MNN
//
//  Created by MNN on 2019/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef CPUMatrixBandPart_hpp
#define CPUMatrixBandPart_hpp

#include "device/cpu/CPUDevice.h"
namespace tars {

class CPUMatrixBandPart : public Operation {
 public:
  CPUMatrixBandPart(Device *backend) : Operation(backend) {
    // Do nothing
  }
  virtual ~CPUMatrixBandPart() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  std::shared_ptr<Tensor> mMask;
};
}  // namespace tars

#endif  // CPUMatrixBandPart_hpp

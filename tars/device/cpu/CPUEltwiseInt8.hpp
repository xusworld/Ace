//
//  CPUEltwiseInt8.hpp
//  MNN
//
//  Created by MNN on 2019/08/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUEltwiseInt8_hpp
#define CPUEltwiseInt8_hpp

#include "core/operation.h"

namespace tars {

class CPUEltwiseInt8 : public Operation {
 public:
  CPUEltwiseInt8(Device *backend, const Op *op);
  virtual ~CPUEltwiseInt8();
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  std::shared_ptr<Tensor> mInput0Scales;
  std::shared_ptr<Tensor> mInput1Scales;
  std::shared_ptr<Tensor> mOutputScales;
  bool isEltwiseInt8 = true;
};

}  // namespace tars

#endif /* CPUEltwiseInt8_hpp */

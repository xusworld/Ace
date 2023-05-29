//
//  CPUSetDiff1D.hpp
//  MNN
//
//  Created by MNN on 2019/6/11.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#ifndef CPUSetDiff1D_hpp
#define CPUSetDiff1D_hpp

#include "device/cpu/CPUDevice.h"
namespace tars {
class CPUSetDiff1D : public Operation {
 public:
  CPUSetDiff1D(Device *bn) : Operation(bn) {
    // Do nothing
  }
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
};
};     // namespace tars
#endif /* CPUSetDiff1D_hpp */

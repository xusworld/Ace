//
//  CPUSelect.hpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#ifndef CPUSelect_hpp
#define CPUSelect_hpp

#include "device/cpu/CPUDevice.h"
namespace tars {
class CPUSelect : public Operation {
 public:
  CPUSelect(Device *bn) : Operation(bn) {
    // Do nothing
  }
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
};
};  // namespace tars

#endif /* CPUSelect_hpp */

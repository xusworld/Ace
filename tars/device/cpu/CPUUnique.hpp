//
//  CPUUnique.hpp
//  MNN
//
//  Created by MNN on 2019/6/11.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#ifndef CPUUnique_hpp
#define CPUUnique_hpp

#include "device/cpu/CPUDevice.h"
namespace tars {
class CPUUnique : public Operation {
 public:
  CPUUnique(Device *bn) : Operation(bn) {
    // Do nothing
  }
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
};
};     // namespace tars
#endif /* CPUUnique_hpp */

//
//  CPUUnravelIndex.hpp
//  MNN
//
//  Created by MNN on 2018/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUUnravelIndex_hpp
#define CPUUnravelIndex_hpp

#include "core/operation.h"

namespace tars {

class CPUUnravelIndex : public Operation {
 public:
  CPUUnravelIndex(Device *b) : Operation(b) {}
  virtual ~CPUUnravelIndex() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
};

}  // namespace tars

#endif /* CPUUnravelIndex_hpp */

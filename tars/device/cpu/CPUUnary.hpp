//
//  CPUUnary.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUUnary_hpp
#define CPUUnary_hpp

#include "compute/CommonOptFunction.h"
#include "core/operation.h"

namespace tars {
class CPUUnary : public Operation {
 public:
  CPUUnary(Device *b, MNNUnaryExecute proc);
  virtual ~CPUUnary() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

  static MNNUnaryExecute selectForFloat(int type, int precision);

 protected:
  MNNUnaryExecute mProc;
};
}  // namespace tars
#endif /* CPUUnary_hpp */

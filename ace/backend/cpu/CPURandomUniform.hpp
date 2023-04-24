//
//  CPURandomUniform.h
//  MNN
//
//  Created by MNN on 2020/8/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPURandomUniform_h
#define CPURandomUniform_h

#include "core/Execution.hpp"

namespace ace {
class CPURandomUniform : public Execution {
 public:
  CPURandomUniform(Backend *b, const ace::Op *op) : ace::Execution(b), mOp(op) {
    // nothing to do
  }
  virtual ~CPURandomUniform() = default;
  virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

 private:
  const ace::Op *mOp;
};

}  // namespace ace

#endif /* CPURandomUniform_h */

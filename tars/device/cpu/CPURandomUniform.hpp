//
//  CPURandomUniform.h
//  MNN
//
//  Created by MNN on 2020/8/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPURandomUniform_h
#define CPURandomUniform_h

#include "core/operation.h"

namespace tars {
class CPURandomUniform : public Operation {
 public:
  CPURandomUniform(Device *b, const tars::Op *op)
      : tars::Operation(b), mOp(op) {
    // nothing to do
  }
  virtual ~CPURandomUniform() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  const tars::Op *mOp;
};

class CPURandomNormal : public Operation {
 public:
  CPURandomNormal(Device *b, const tars::Op *op) : tars::Operation(b), mOp(op) {
    // nothing to do
  }
  virtual ~CPURandomNormal() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  const tars::Op *mOp;
};

}  // namespace tars

#endif /* CPURandomUniform_h */

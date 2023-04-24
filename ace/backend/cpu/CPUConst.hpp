//
//  CPUConst.hpp
//  MNN
//
//  Created by MNN on 2018/08/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUConst_hpp
#define CPUConst_hpp

#include "core/Execution.hpp"

namespace ace {
class CPUConst : public Execution {
 public:
  CPUConst(Backend *b, const ace::Op *op);
  virtual ~CPUConst() = default;
  virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

 protected:
  const ace::Op *mOp;
};
}  // namespace ace
#endif /* CPUConst_hpp */

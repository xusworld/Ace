//
//  CPUBinary.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUBinary_hpp
#define CPUBinary_hpp

#include "compute/CommonOptFunction.h"
#include "core/Execution.hpp"
namespace ace {
class CPUBinary : public Execution {
 public:
  CPUBinary(Backend *b, MNNBinaryExecute proc) : Execution(b) { mProc = proc; }
  virtual ~CPUBinary() = default;
  virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

  static MNNBinaryExecute selectForFloat(int opType);

 private:
  MNNBinaryExecute mProc;
  int mNeedBroadcastIndex = -1;
  int mTotalSize;
};
}  // namespace ace
#endif /* CPUBinary_hpp */

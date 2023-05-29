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
#include "core/operation.h"
#include "device/cpu/CPURelu.hpp"
namespace tars {
class CPUBinary : public Operation {
 public:
  CPUBinary(Device *b, MNNBinaryExecute proc, int activationType)
      : Operation(b) {
    mProc = proc;
    mActivationType = activationType;
  }
  virtual ~CPUBinary() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

  static MNNBinaryExecute selectForFloat(int opType);
  static MNNBinaryExecute selectForInt(int opType);

 private:
  MNNBinaryExecute mProc;
  int mNeedBroadcastIndex = -1;
  int mTotalSize;
  int mActivationType = 0;
  std::shared_ptr<Operation> mActivationExe;
};
}  // namespace tars
#endif /* CPUBinary_hpp */

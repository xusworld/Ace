//
//  CPURNNSequenceGRU.hpp
//  MNN
//
//  Created by MNN on 2019/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPURNNSequenceGRU_hpp
#define CPURNNSequenceGRU_hpp

#include "core/operation.h"

namespace tars {

class CPURNNSequenceGRU : public Operation {
 public:
  CPURNNSequenceGRU(const Op *op, Device *backend);
  virtual ~CPURNNSequenceGRU();
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;

 private:
  bool mKeepAllOutputs;
  bool mIsBidirectionalRNN;
  bool mlinearBeforeReset;
  int mNumUnits;

  std::shared_ptr<Tensor> mHiddenState;
  std::shared_ptr<Tensor> mInputAndState;
  std::shared_ptr<Tensor> mGate;
  std::shared_ptr<Tensor> mResetHt;
};

}  // namespace tars

#endif /* CPURNNSequenceGRU_hpp */

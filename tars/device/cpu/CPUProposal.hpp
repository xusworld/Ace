//
//  CPUProposal.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUProposal_hpp
#define CPUProposal_hpp

#include <functional>

#include "MNN_generated.h"
#include "core/AutoStorage.h"
#include "core/operation.h"

namespace tars {

class CPUProposal : public Operation {
 public:
  CPUProposal(Device *backend, const Proposal *proposal);
  virtual ~CPUProposal() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  const Proposal *mProposal;
  AutoStorage<float> mAnchors;
  Tensor mScore;
  std::function<void()> mRun;
};

}  // namespace tars

#endif /* CPUProposal_hpp */

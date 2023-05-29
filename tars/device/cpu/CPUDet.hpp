//
//  CPUDet.hpp
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDet_hpp
#define CPUDet_hpp

#include "core/operation.h"
#include "core/tensor.h"

namespace tars {
class CPUDet : public Operation {
 public:
  CPUDet(Device *bn) : Operation(bn) {}
  virtual ~CPUDet() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  std::shared_ptr<Tensor> mTempMat, mTempRowPtrs;
};

}  // namespace tars
#endif /* CPUDet_hpp */

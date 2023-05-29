//
//  CPUScale.hpp
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUScale_hpp
#define CPUScale_hpp

#include "core/operation.h"
#include "core/tensor.h"

namespace tars {
class CPUScale : public Operation {
 public:
  CPUScale(const Op *op, Device *bn);
  virtual ~CPUScale();
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  std::shared_ptr<Tensor> mScaleBias;
};

}  // namespace tars
#endif /* CPUScale_hpp */

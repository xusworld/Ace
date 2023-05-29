//
//  CPUInt8ToFloat.hpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUInt8ToFloat_hpp
#define CPUInt8ToFloat_hpp

#include "core/operation.h"
#include "core/tensor.h"

namespace tars {

class CPUInt8ToFloat : public Operation {
 public:
  CPUInt8ToFloat(Device *backend, const tars::Op *param);
  virtual ~CPUInt8ToFloat();
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  std::shared_ptr<Tensor> mScales;

  bool mSingle = false;
  int8_t mZeroPoint;
};

}  // namespace tars

#endif /* CPUInt8ToFloat_hpp */

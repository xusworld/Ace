//
//  CPUInt8ToFloat.hpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUInt8ToFloat_hpp
#define CPUInt8ToFloat_hpp

#include <ace/tensor.h>

#include "core/Execution.hpp"

namespace ace {

class CPUInt8ToFloat : public Execution {
 public:
  CPUInt8ToFloat(Backend *backend, const ace::Op *param);
  virtual ~CPUInt8ToFloat();
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

 private:
  std::shared_ptr<Tensor> mScales;

  bool mSingle = false;
  int8_t mZeroPoint;
};

}  // namespace ace

#endif /* CPUInt8ToFloat_hpp */

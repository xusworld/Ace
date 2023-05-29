//
//  CPUQuantizedLogistic.hpp
//  MNN
//
//  Created by MNN on 2018/12/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUQuantizedLogistic_hpp
#define CPUQuantizedLogistic_hpp

#include "TFQuantizeOp_generated.h"
#include "core/operation.h"

namespace tars {

class CPUQuantizedLogistic : public Operation {
 public:
  CPUQuantizedLogistic(Device *backend, const Op *op);
  virtual ~CPUQuantizedLogistic() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  const QuantizedLogistic *mLogisticParam;
  int mInputMultiplier;
  int mInputZeroPoint;
  int mInputLeftShift;
  int mInputRangeRadius;
};

}  // namespace tars
#endif /* CPUQuantizedLogistic_hpp */

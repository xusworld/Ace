//
//  CPUDequantize.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDequantize_hpp
#define CPUDequantize_hpp

#include "TFQuantizeOp_generated.h"
#include "core/operation.h"

namespace tars {

template <typename T>
class CPUDequantize : public Operation {
 public:
  CPUDequantize(Device *backend, QuantizeMode mode, const Op *op);
  virtual ~CPUDequantize() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  float mHalfRange;
  QuantizeMode mMode;
  bool mIsLiteDequantize;
  int mZeroPoint;
  float mScale;
};

}  // namespace tars

#endif /* CPUDequantize_hpp */

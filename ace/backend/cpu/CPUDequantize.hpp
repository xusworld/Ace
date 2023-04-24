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
#include "core/Execution.hpp"

namespace ace {

template <typename T>
class CPUDequantize : public Execution {
 public:
  CPUDequantize(Backend *backend, QuantizeMode mode, const Op *op);
  virtual ~CPUDequantize() = default;
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

 private:
  float mHalfRange;
  QuantizeMode mMode;
  bool mIsLiteDequantize;
  int mZeroPoint;
  float mScale;
};

}  // namespace ace

#endif /* CPUDequantize_hpp */

//
//  ShapeDequantize.cpp
//  MNN
//
//  Created by MNN on 2019/01/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "shape/SizeComputer.hpp"

namespace tars {
class ShapeDequantize : public SizeComputer {
  virtual bool onComputeSize(
      const tars::Op *op, const std::vector<Tensor *> &inputs,
      const std::vector<Tensor *> &outputs) const override {
    MNN_ASSERT(3 == inputs.size() || 1 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    TensorUtils::copyShape(inputs[0], outputs[0], true);
    outputs[0]->buffer().type = halide_type_of<float>();
    return true;
  }
};
REGISTER_SHAPE(ShapeDequantize, OpType_Dequantize);
}  // namespace tars

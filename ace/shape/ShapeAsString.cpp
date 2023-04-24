//
//  ShapeAsString.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "shape/SizeComputer.hpp"

namespace ace {
class AsStringComputer : public SizeComputer {
 public:
  virtual bool onComputeSize(
      const ace::Op* op, const std::vector<Tensor*>& inputs,
      const std::vector<Tensor*>& outputs) const override {
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(1 == outputs.size());

    auto output = outputs[0];
    auto input = inputs[0];
    TensorUtils::copyShape(input, output, true);

    output->setType(ace::DataType_DT_STRING);

    return true;
  }
};

REGISTER_SHAPE(AsStringComputer, OpType_AsString);

}  // namespace ace

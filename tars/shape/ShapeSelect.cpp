//
//  ShapeSelect.cpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "shape/SizeComputer.hpp"
namespace tars {

class SelectSizeComputer : public SizeComputer {
 public:
  virtual bool onComputeSize(
      const tars::Op* op, const std::vector<Tensor*>& inputs,
      const std::vector<Tensor*>& outputs) const override {
    MNN_ASSERT(3 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    const auto& ib = inputs[1]->buffer();
    auto& ob = outputs[0]->buffer();
    ob.type = inputs[1]->buffer().type;
    bool res = SizeComputer::computeBroadCastDims(op, inputs, outputs);
    if (!res) {
      return false;
    }
    TensorUtils::getDescribe(outputs[0])->dimensionFormat =
        TensorUtils::getDescribe(inputs[1])->dimensionFormat;
    return true;
  }
};

REGISTER_SHAPE(SelectSizeComputer, OpType_Select);
}  // namespace tars

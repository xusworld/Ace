//
//  ShapeReduceJoin.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "shape/SizeComputer.hpp"

namespace ace {
class ReduceJoinComputer : public SizeComputer {
 public:
  virtual bool onComputeSize(
      const ace::Op* op, const std::vector<Tensor*>& inputs,
      const std::vector<Tensor*>& outputs) const override {
    MNN_ASSERT(2 == inputs.size());
    MNN_ASSERT(1 == outputs.size());

    auto output = outputs[0];
    auto input = inputs[0];
    auto axis = inputs[1];

    // support reduce 1 dimension, only
    MNN_ASSERT(axis->size() == axis->buffer().type.bytes());

    MNN_ASSERT(axis->host<int32_t>()[0] >= 0);
    std::vector<int> shape;
    for (int i = 0; i < input->buffer().dimensions; i++) {
      if (i != axis->host<int32_t>()[0]) {
        shape.push_back(input->buffer().dim[i].extent);
      } else {
        if (op->main_as_ReduceJoin()->keepDims()) {
          shape.push_back(1);
        }
      }
    }
    output->buffer().dimensions = (int)shape.size();
    for (int i = 0; i < shape.size(); i++) {
      output->buffer().dim[i].extent = shape[i];
    }
    output->setType(DataType_DT_STRING);
    TensorUtils::getDescribe(outputs[0])->dimensionFormat = DATA_FORMAT_NHWC;
    return true;
  }
};

REGISTER_SHAPE_INPUTS(ReduceJoinComputer, OpType_ReduceJoin, {1});
}  // namespace ace
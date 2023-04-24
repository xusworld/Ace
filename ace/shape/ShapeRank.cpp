//
//  ShapeRank.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/TensorUtils.hpp"
#include "shape/SizeComputer.hpp"
namespace ace {

class RankComputer : public SizeComputer {
  virtual bool onComputeSize(
      const ace::Op* op, const std::vector<Tensor*>& inputs,
      const std::vector<Tensor*>& outputs) const override {
    // output is Scalar
    outputs[0]->buffer().dimensions = 0;
    outputs[0]->setType(ace::DataType_DT_INT32);
    TensorUtils::getDescribe(outputs[0])->dimensionFormat = DATA_FORMAT_NHWC;
    return true;
  }
};

REGISTER_SHAPE(RankComputer, OpType_Rank);
}  // namespace ace

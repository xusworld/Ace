//
//  ShapeSpaceToDepth.cpp
//  MNN
//
//  Created by MNN on 2019/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "shape/SizeComputer.hpp"

namespace tars {

class SpaceToDepthSizeComputer : public SizeComputer {
  virtual bool onComputeSize(
      const tars::Op* op, const std::vector<Tensor*>& inputs,
      const std::vector<Tensor*>& outputs) const override {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    MNN_ASSERT(inputs[0]->buffer().dimensions == 4);

    // here only implement NHWC
    // TODO: implement NC4HW4
    const int blockSize = op->main_as_DepthSpaceParam()->blockSize();
    MNN_ASSERT(blockSize > 1);
    MNN_ASSERT(inputs[0]->buffer().dim[1].extent % blockSize == 0);
    MNN_ASSERT(inputs[0]->buffer().dim[2].extent % blockSize == 0);

    auto& ib = inputs[0]->buffer();
    auto& ob = outputs[0]->buffer();

    ob.dimensions = ib.dimensions;
    ob.type = ib.type;
    auto format = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
    ob.dim[0].extent = ib.dim[0].extent;
    if (MNN_DATA_FORMAT_NHWC == format) {
      ob.dim[1].extent = ib.dim[1].extent / blockSize;
      ob.dim[2].extent = ib.dim[2].extent / blockSize;
      ob.dim[3].extent = ib.dim[3].extent * (blockSize * blockSize);
    } else {
      ob.dim[3].extent = ib.dim[3].extent / blockSize;
      ob.dim[2].extent = ib.dim[2].extent / blockSize;
      ob.dim[1].extent = ib.dim[1].extent * (blockSize * blockSize);
    }
    TensorUtils::getDescribe(outputs[0])->dimensionFormat =
        TensorUtils::getDescribe(inputs[0])->dimensionFormat;

    return true;
  }
};

REGISTER_SHAPE(SpaceToDepthSizeComputer, OpType_SpaceToDepth);

}  // namespace tars

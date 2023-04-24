//
//  ShapeTopKV2.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "shape/SizeComputer.hpp"

namespace ace {

class TopKV2SizeComputer : public SizeComputer {
  virtual bool onComputeSize(
      const ace::Op* op, const std::vector<Tensor*>& inputs,
      const std::vector<Tensor*>& outputs) const override {
    MNN_ASSERT(2 == inputs.size());
    MNN_ASSERT(2 == outputs.size());
    auto input = inputs[0];
    auto kTensor = inputs[1];
    MNN_ASSERT(kTensor->buffer().dimensions == 0);  // Scalar
    MNN_ASSERT(kTensor->getType().code == halide_type_int);
    const int k = kTensor->host<int32_t>()[0];
    const int inputDimension = input->buffer().dimensions;
    // outputs: 0 --> data, 1 --> index

    auto outputData = outputs[0];
    outputData->buffer().dimensions = inputDimension;
    memcpy(outputData->buffer().dim, input->buffer().dim,
           inputDimension * sizeof(halide_dimension_t));
    outputData->buffer().dim[inputDimension - 1].extent = k;
    outputData->buffer().type = input->buffer().type;

    auto outputIndices = outputs[1];
    outputIndices->buffer().dimensions = inputDimension;
    memcpy(outputIndices->buffer().dim, input->buffer().dim,
           inputDimension * sizeof(halide_dimension_t));
    outputIndices->buffer().dim[inputDimension - 1].extent = k;
    outputIndices->setType(ace::DataType_DT_INT32);
    TensorUtils::getDescribe(outputs[0])->dimensionFormat =
        TensorUtils::getDescribe(inputs[0])->dimensionFormat;
    TensorUtils::getDescribe(outputs[1])->dimensionFormat =
        TensorUtils::getDescribe(inputs[1])->dimensionFormat;

    return true;
  }
};

REGISTER_SHAPE_INPUTS(TopKV2SizeComputer, OpType_TopKV2, {1});
}  // namespace ace

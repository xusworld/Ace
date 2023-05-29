//
//  ReduceOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ReduceOnnx);

tars::OpType ReduceOnnx::opType() { return tars::OpType_Reduction; }
tars::OpParameter ReduceOnnx::type() {
  return tars::OpParameter_ReductionParam;
}

void ReduceOnnx::run(tars::OpT *dstOp, const onnx::NodeProto *onnxNode,
                     OnnxScope *scope) {
  auto param = new tars::ReductionParamT;

  std::vector<int> axes;
  bool keepdims = true;
  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();

    if (attributeName == "axes") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS)
          << "Node Attribute ERROR";
      const int size = attributeProto.ints_size();
      for (int k = 0; k < size; ++k) {
        axes.push_back(attributeProto.ints(k));
      }
    } else if (attributeName == "keepdims") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT)
          << "Node Attribute ERROR";
      keepdims = static_cast<bool>(attributeProto.i());
    }
  }

  auto type = onnxNode->op_type();
  if (type == "ReduceMean") {
    param->operation = tars::ReductionType_MEAN;
  } else if (type == "ReduceMax") {
    param->operation = tars::ReductionType_MAXIMUM;
  } else if (type == "ReduceMin") {
    param->operation = tars::ReductionType_MINIMUM;
  } else if (type == "ReduceProd") {
    param->operation = tars::ReductionType_PROD;
  } else if (type == "ReduceSum") {
    param->operation = tars::ReductionType_SUM;
  } else {
    DLOG(ERROR) << "TODO ==> " << type;
  }

  param->dType = tars::DataType_DT_FLOAT;
  param->dim = axes;
  param->keepDims = keepdims;
  dstOp->main.value = param;
}

REGISTER_CONVERTER(ReduceOnnx, ReduceMean);
REGISTER_CONVERTER(ReduceOnnx, ReduceMax);
REGISTER_CONVERTER(ReduceOnnx, ReduceMin);
REGISTER_CONVERTER(ReduceOnnx, ReduceProd);
REGISTER_CONVERTER(ReduceOnnx, ReduceSum);

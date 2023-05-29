//
//  CastOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(CastOnnx);

tars::OpType CastOnnx::opType() { return tars::OpType_Cast; }
tars::OpParameter CastOnnx::type() { return tars::OpParameter_CastParam; }

void CastOnnx::run(tars::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope *scope) {
  std::unique_ptr<tars::CastParamT> castParam(new tars::CastParamT);

  // not to use srcT parameter!
  castParam->srcT = tars::DataType_MAX;

  ::onnx::TensorProto_DataType castTo = ::onnx::TensorProto_DataType_UNDEFINED;
  const int attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "to") {
      castTo = static_cast<::onnx::TensorProto_DataType>(attributeProto.i());
    }
  }

  castParam->dstT = onnxOpConverter::convertDataType(castTo);
  dstOp->main.value = castParam.release();
}

REGISTER_CONVERTER(CastOnnx, Cast);

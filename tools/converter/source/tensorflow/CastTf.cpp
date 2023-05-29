//
//  CastTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(CastTf);

tars::OpType CastTf::opType() { return tars::OpType_Cast; }
tars::OpParameter CastTf::type() { return tars::OpParameter_CastParam; }

void CastTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto parameter = new tars::CastParamT;
  tensorflow::AttrValue value;
  parameter->dstT = tars::DataType_DT_INVALID;
  parameter->srcT = tars::DataType_DT_INVALID;
  if (find_attr_value(srcNode->tfNode, "DstT", value)) {
    parameter->dstT = (tars::DataType)value.type();
  }
  if (find_attr_value(srcNode->tfNode, "SrcT", value)) {
    parameter->srcT = (tars::DataType)value.type();
  }
  DCHECK(parameter->srcT != tars::DataType_DT_INVALID &&
         parameter->dstT != tars::DataType_DT_INVALID)
      << "Cast Parameter ERROR!!! ===> " << srcNode->opName;

  dstOp->main.value = parameter;
}

REGISTER_CONVERTER(CastTf, Cast);

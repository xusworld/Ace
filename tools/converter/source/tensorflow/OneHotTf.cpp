//
//  OneHotTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(OneHotTf);

tars::OpType OneHotTf::opType() { return tars::OpType_OneHot; }

tars::OpParameter OneHotTf::type() { return tars::OpParameter_OneHotParam; }

void OneHotTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto param = new tars::OneHotParamT;

  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "T", value)) {
    param->dType = static_cast<tars::DataType>(value.type());
  }

  if (find_attr_value(srcNode->tfNode, "axis", value)) {
    param->axis = value.i();
  }

  dstOp->main.value = param;
}

REGISTER_CONVERTER(OneHotTf, OneHot);

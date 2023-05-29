//
//  UnpackTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(UnpackTf);

tars::OpType UnpackTf::opType() { return tars::OpType_Unpack; }
tars::OpParameter UnpackTf::type() { return tars::OpParameter_Axis; }

void UnpackTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto axisT = new tars::AxisT;
  tensorflow::AttrValue value;
  axisT->axis = 1;  // default
  find_attr_value(srcNode->tfNode, "axis", value);
  axisT->axis = value.i();
  dstOp->main.value = axisT;
}

REGISTER_CONVERTER(UnpackTf, Unpack);

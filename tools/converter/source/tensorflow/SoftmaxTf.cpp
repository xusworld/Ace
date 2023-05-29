//
//  SoftmaxTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(SoftmaxTf);

tars::OpType SoftmaxTf::opType() { return tars::OpType_Softmax; }
tars::OpParameter SoftmaxTf::type() { return tars::OpParameter_Axis; }

void SoftmaxTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto axisT = new tars::AxisT;
  axisT->axis = -1;
  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "axis", value)) {
    axisT->axis = value.i();
  }
  dstOp->main.value = axisT;
}

REGISTER_CONVERTER(SoftmaxTf, Softmax);

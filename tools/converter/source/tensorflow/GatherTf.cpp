//
//  GatherTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(GatherTf);

tars::OpType GatherTf::opType() { return tars::OpType_Gather; }
tars::OpParameter GatherTf::type() { return tars::OpParameter_Gather; }

void GatherTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto parameter = new tars::GatherT;
  parameter->axis = 1;

  tensorflow::AttrValue value;

  find_attr_value(srcNode->tfNode, "Tindices", value);
  parameter->Tindices = (tars::DataType)value.type();

  find_attr_value(srcNode->tfNode, "Tparams", value);
  parameter->Tparams = (tars::DataType)value.type();

  dstOp->main.value = parameter;
}

REGISTER_CONVERTER(GatherTf, Gather);

DECLARE_OP_CONVERTER(GatherNDTf);
tars::OpType GatherNDTf::opType() { return tars::OpType_GatherND; }
tars::OpParameter GatherNDTf::type() { return tars::OpParameter_NONE; }
void GatherNDTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  // Do nothing
}
REGISTER_CONVERTER(GatherNDTf, GatherNd);

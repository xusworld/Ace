//
//  ReduceJoinTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(ReduceJoinTf);

tars::OpType ReduceJoinTf::opType() { return tars::OpType_ReduceJoin; }
tars::OpParameter ReduceJoinTf::type() { return tars::OpParameter_ReduceJoin; }

void ReduceJoinTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto parameter = new tars::ReduceJoinT;

  tensorflow::AttrValue value;

  find_attr_value(srcNode->tfNode, "keep_dims", value);
  parameter->keepDims = value.b();

  find_attr_value(srcNode->tfNode, "separator", value);
  parameter->separator = value.s();

  dstOp->main.value = parameter;
}

// REGISTER_CONVERTER(ReduceJoinTf, ReduceJoin);

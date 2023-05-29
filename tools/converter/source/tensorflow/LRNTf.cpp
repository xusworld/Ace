//
//  LRNTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(LRNTf);

tars::OpType LRNTf::opType() { return tars::OpType_LRN; }
tars::OpParameter LRNTf::type() { return tars::OpParameter_LRN; }

void LRNTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto lrnParam = new tars::LRNT;
  lrnParam->regionType = 0;
  tensorflow::AttrValue value;

  find_attr_value(srcNode->tfNode, "alpha", value);
  lrnParam->alpha = value.f();

  find_attr_value(srcNode->tfNode, "beta", value);
  lrnParam->beta = value.f();

  find_attr_value(srcNode->tfNode, "depth_radius", value);
  lrnParam->localSize = 2 * value.i() + 1;

  dstOp->main.value = lrnParam;
}

REGISTER_CONVERTER(LRNTf, LRN);

//
//  RandomUniformTf.cpp
//  MNNConverter
//
//  Created by MNN on 2020/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

using namespace tars;
DECLARE_OP_CONVERTER(RandomUniformTf);

tars::OpType RandomUniformTf::opType() { return tars::OpType_RandomUniform; }
tars::OpParameter RandomUniformTf::type() {
  return tars::OpParameter_RandomUniform;
}

void RandomUniformTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto parameter = new tars::RandomUniformT;
  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "seed", value)) {
    parameter->seed = value.i();
  }
  if (find_attr_value(srcNode->tfNode, "seed2", value)) {
    parameter->seed2 = value.i();
  }
  if (find_attr_value(srcNode->tfNode, "type", value)) {
    parameter->type = static_cast<tars::DataType>(value.i());
  }
  dstOp->main.value = parameter;
}

REGISTER_CONVERTER(RandomUniformTf, RandomUniform);

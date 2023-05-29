//
//  WhereTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(WhereTf);

tars::OpType WhereTf::opType() { return tars::OpType_Where; }
tars::OpParameter WhereTf::type() { return tars::OpParameter_Extra; }

void WhereTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  // for distinguish old-version
  auto parameter = new tars::ExtraT;
  parameter->engine = "tensorflow";
  parameter->type = "control_flow_where";
  dstOp->main.value = parameter;
}

REGISTER_CONVERTER(WhereTf, Where);

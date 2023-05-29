//
//  BroadcastToTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(BroadcastToTf);

tars::OpType BroadcastToTf::opType() { return tars::OpType_BroadcastTo; }
tars::OpParameter BroadcastToTf::type() { return tars::OpParameter_NONE; }

void BroadcastToTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(BroadcastToTf, BroadcastTo);

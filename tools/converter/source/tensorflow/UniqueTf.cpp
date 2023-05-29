//
//  ListDiff.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(Unique);

tars::OpType Unique::opType() { return tars::OpType_Unique; }
tars::OpParameter Unique::type() { return tars::OpParameter_NONE; }

void Unique::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(Unique, Unique);

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

DECLARE_OP_CONVERTER(ListDiff);

tars::OpType ListDiff::opType() { return tars::OpType_SetDiff1D; }
tars::OpParameter ListDiff::type() { return tars::OpParameter_NONE; }

void ListDiff::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(ListDiff, ListDiff);
REGISTER_CONVERTER(ListDiff, SetDiff1d);

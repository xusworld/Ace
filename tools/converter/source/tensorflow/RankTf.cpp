//
//  RankTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(Rank);

tars::OpType Rank::opType() { return tars::OpType_Rank; }
tars::OpParameter Rank::type() { return tars::OpParameter_NONE; }

void Rank::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(Rank, Rank);

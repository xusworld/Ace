//
//  SigmoidTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(SigmoidTf);

tars::OpType SigmoidTf::opType() { return tars::OpType_Sigmoid; }
tars::OpParameter SigmoidTf::type() { return tars::OpParameter_NONE; }

void SigmoidTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(SigmoidTf, Sigmoid);

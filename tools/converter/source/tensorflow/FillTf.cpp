//
//  FillTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(FillTf);

tars::OpType FillTf::opType() { return tars::OpType_Fill; }
tars::OpParameter FillTf::type() { return tars::OpParameter_Fill; }

void FillTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = nullptr;
}
REGISTER_CONVERTER(FillTf, Fill);

DECLARE_OP_CONVERTER(ZerosLikeTf);
tars::OpType ZerosLikeTf::opType() { return tars::OpType_ZerosLike; }
tars::OpParameter ZerosLikeTf::type() { return tars::OpParameter_NONE; }

void ZerosLikeTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(ZerosLikeTf, ZerosLike);

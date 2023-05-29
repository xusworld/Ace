//
//  ScatterNdTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(ScatterNdTf);

tars::OpType ScatterNdTf::opType() { return tars::OpType_ScatterNd; }
tars::OpParameter ScatterNdTf::type() { return tars::OpParameter_NONE; }

void ScatterNdTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(ScatterNdTf, ScatterNd);

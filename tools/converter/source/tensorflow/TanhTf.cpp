//
//  TanhTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(TanhTf);

tars::OpType TanhTf::opType() { return tars::OpType_TanH; }
tars::OpParameter TanhTf::type() { return tars::OpParameter_NONE; }

void TanhTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(TanhTf, Tanh);

//
//  TileTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(TileTf);

tars::OpType TileTf::opType() { return tars::OpType_Tile; }
tars::OpParameter TileTf::type() { return tars::OpParameter_NONE; }

void TileTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(TileTf, Tile);

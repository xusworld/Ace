//
//  EluTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(EluTf);

tars::OpType EluTf::opType() { return tars::OpType_ELU; }
tars::OpParameter EluTf::type() { return tars::OpParameter_ELU; }

void EluTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto elu = new tars::ELUT;
  elu->alpha = 1.0f;
  dstOp->main.value = elu;
}

REGISTER_CONVERTER(EluTf, Elu);

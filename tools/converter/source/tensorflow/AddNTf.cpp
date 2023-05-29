//
//  AddNTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <map>
#include <string>

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

using namespace tars;

DECLARE_OP_CONVERTER(AddNTf);

tars::OpType AddNTf::opType() { return tars::OpType_Eltwise; }

tars::OpParameter AddNTf::type() { return tars::OpParameter_Eltwise; }

void AddNTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto elt = new tars::EltwiseT;
  dstOp->main.value = elt;
  elt->type = tars::EltwiseType_SUM;
}

REGISTER_CONVERTER(AddNTf, AddN);
REGISTER_CONVERTER(AddNTf, AccumulateNV2);

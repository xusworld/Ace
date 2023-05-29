//
//  LinSpaceTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <map>
#include <string>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

using namespace tars;

DECLARE_OP_CONVERTER(LinSpaceTf);

tars::OpType LinSpaceTf::opType() { return tars::OpType_LinSpace; }

tars::OpParameter LinSpaceTf::type() { return tars::OpParameter_NONE; }

void LinSpaceTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(LinSpaceTf, LinSpace);

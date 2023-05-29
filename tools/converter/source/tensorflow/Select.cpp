//
//  Select.cpp
//  MNNConverter
//
//  Created by MNN on 2019/05/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(SelectTf);

tars::OpType SelectTf::opType() { return tars::OpType_Select; }
tars::OpParameter SelectTf::type() { return tars::OpParameter_NONE; }
void SelectTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  // Do nothing
}
REGISTER_CONVERTER(SelectTf, SelectV2);

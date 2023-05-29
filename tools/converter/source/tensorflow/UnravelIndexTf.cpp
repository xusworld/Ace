//
//  UnravelIndexTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(UnravelIndexTf);

tars::OpType UnravelIndexTf::opType() { return tars::OpType_UnravelIndex; }

tars::OpParameter UnravelIndexTf::type() { return tars::OpParameter_NONE; }

void UnravelIndexTf::run(tars::OpT *dstOp, TmpNode *srcNode) { return; }

REGISTER_CONVERTER(UnravelIndexTf, UnravelIndex);

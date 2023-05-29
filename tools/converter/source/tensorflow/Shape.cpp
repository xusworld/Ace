//
//  Shape.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(ShapeTf);

tars::OpType ShapeTf::opType() { return tars::OpType_Shape; }
tars::OpParameter ShapeTf::type() { return tars::OpParameter_NONE; }

void ShapeTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(ShapeTf, Shape);

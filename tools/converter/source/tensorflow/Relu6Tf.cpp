//
//  Relu6Tf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(Relu6Tf);

tars::OpType Relu6Tf::opType() { return tars::OpType_ReLU6; }
tars::OpParameter Relu6Tf::type() { return tars::OpParameter_Relu6; }

void Relu6Tf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto relu6 = new tars::Relu6T;
  dstOp->main.value = relu6;
}

REGISTER_CONVERTER(Relu6Tf, Relu6);

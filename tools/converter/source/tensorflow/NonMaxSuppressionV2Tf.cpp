//
//  NonMaxSuppressionV2Tf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(NonMaxSuppressionV2);

tars::OpType NonMaxSuppressionV2::opType() {
  return tars::OpType_NonMaxSuppressionV2;
}
tars::OpParameter NonMaxSuppressionV2::type() {
  return tars::OpParameter_NonMaxSuppressionV2;
}

void NonMaxSuppressionV2::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(NonMaxSuppressionV2, NonMaxSuppressionV2);
REGISTER_CONVERTER(NonMaxSuppressionV2, NonMaxSuppressionV3);

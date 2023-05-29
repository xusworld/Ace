//
//  ReverseSequence.cpp
//  MNNConverter
//
//  Created by MNN on 2019/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(ReverseSequence);

tars::OpType ReverseSequence::opType() { return tars::OpType_ReverseSequence; }
tars::OpParameter ReverseSequence::type() {
  return tars::OpParameter_ReverseSequenceParam;
}

void ReverseSequence::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto param = new tars::ReverseSequenceParamT;
  param->batchDim = 0;
  param->seqDim = 0;
  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "batch_dim", value)) {
    param->batchDim = value.i();
  }
  if (find_attr_value(srcNode->tfNode, "seq_dim", value)) {
    param->seqDim = value.i();
  }
  dstOp->main.value = param;
}

REGISTER_CONVERTER(ReverseSequence, ReverseSequence);

DECLARE_OP_CONVERTER(Reverse);

tars::OpType Reverse::opType() { return tars::OpType_Reverse; }
tars::OpParameter Reverse::type() { return tars::OpParameter_NONE; }

void Reverse::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(Reverse, ReverseV2);

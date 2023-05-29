//
//  LSTMBlockCellTf.cpp
//  MNNConverter
//
//  Created by MNN on 2021/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(LSTMBlockCellTf);

tars::OpType LSTMBlockCellTf::opType() { return tars::OpType_LSTMBlockCell; }
tars::OpParameter LSTMBlockCellTf::type() {
  return tars::OpParameter_LSTMBlockCell;
}

void LSTMBlockCellTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto lstmParam = new tars::LSTMBlockCellT;
  tensorflow::AttrValue value;

  if (find_attr_value(srcNode->tfNode, "cell_clip", value)) {
    lstmParam->cell_clip = value.f();
  }

  if (find_attr_value(srcNode->tfNode, "forget_bias", value)) {
    lstmParam->forget_bias = value.f();
  }

  if (find_attr_value(srcNode->tfNode, "use_peephole", value)) {
    lstmParam->use_peephole = value.b();
  }
  dstOp->main.value = lstmParam;
}

REGISTER_CONVERTER(LSTMBlockCellTf, LSTMBlockCell);

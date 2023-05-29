//
//  TransposeTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(Transpose);

tars::OpType Transpose::opType() { return tars::OpType_Transpose; }
tars::OpParameter Transpose::type() { return tars::OpParameter_Transpose; }

void Transpose::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto Transpose = new tars::TransposeT;
  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "Tperm", value)) {
    Transpose->Tperm = (tars::DataType)value.type();
  }
  dstOp->main.value = Transpose;
}

REGISTER_CONVERTER(Transpose, Transpose);

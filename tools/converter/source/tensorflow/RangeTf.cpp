//
//  RangeTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(Range);

tars::OpType Range::opType() { return tars::OpType_Range; }
tars::OpParameter Range::type() { return tars::OpParameter_Range; }

void Range::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto Range = new tars::RangeT;
  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "Tidx", value)) {
    Range->Tidx = (tars::DataType)value.type();
  }
  dstOp->main.value = Range;
}

REGISTER_CONVERTER(Range, Range);

//
//  SliceTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(SliceTf);

tars::OpType SliceTf::opType() { return tars::OpType_SliceTf; }
tars::OpParameter SliceTf::type() { return tars::OpParameter_SliceTf; }

void SliceTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto sliceParam = new tars::SliceTfT;
  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "T", value)) {
    sliceParam->T = (tars::DataType)value.type();
  }
  dstOp->main.value = sliceParam;
}

REGISTER_CONVERTER(SliceTf, Slice);

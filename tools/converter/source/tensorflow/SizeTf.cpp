//
//  SizeTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(Size);

tars::OpType Size::opType() { return tars::OpType_Size; }
tars::OpParameter Size::type() { return tars::OpParameter_Size; }

void Size::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto Size = new tars::SizeT;
  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "out_type", value)) {
    Size->outputDataType = (tars::DataType)value.type();
  }
  dstOp->main.value = Size;
}

REGISTER_CONVERTER(Size, Size);

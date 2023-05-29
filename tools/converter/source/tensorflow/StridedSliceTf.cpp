//
//  StridedSliceTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(StridedSliceTf);

tars::OpType StridedSliceTf::opType() { return tars::OpType_StridedSlice; }
tars::OpParameter StridedSliceTf::type() {
  return tars::OpParameter_StridedSliceParam;
}

void StridedSliceTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto stridedslice = new tars::StridedSliceParamT;

  tensorflow::AttrValue value;
  find_attr_value(srcNode->tfNode, "begin_mask", value);
  stridedslice->beginMask = value.i();

  find_attr_value(srcNode->tfNode, "end_mask", value);
  stridedslice->endMask = value.i();

  find_attr_value(srcNode->tfNode, "ellipsis_mask", value);
  stridedslice->ellipsisMask = value.i();

  find_attr_value(srcNode->tfNode, "new_axis_mask", value);
  stridedslice->newAxisMask = value.i();

  find_attr_value(srcNode->tfNode, "shrink_axis_mask", value);
  stridedslice->shrinkAxisMask = value.i();

  find_attr_value(srcNode->tfNode, "Index", value);
  stridedslice->Index = (tars::DataType)value.type();

  find_attr_value(srcNode->tfNode, "T", value);
  stridedslice->T = (tars::DataType)value.type();

  dstOp->main.value = stridedslice;
}

REGISTER_CONVERTER(StridedSliceTf, StridedSlice);

//
//  CropAndResizeTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(CropAndResize);

tars::OpType CropAndResize::opType() { return tars::OpType_CropAndResize; }
tars::OpParameter CropAndResize::type() {
  return tars::OpParameter_CropAndResize;
}

void CropAndResize::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto CropAndResize = new tars::CropAndResizeT;
  tensorflow::AttrValue value;

  if (find_attr_value(srcNode->tfNode, "extrapolation_value", value)) {
    CropAndResize->extrapolationValue = value.f();
  }

  if (find_attr_value(srcNode->tfNode, "method", value)) {
    if (value.s() == "bilinear") {
      CropAndResize->method = tars::CropAndResizeMethod_BILINEAR;
    } else {
      CropAndResize->method = tars::CropAndResizeMethod_NEAREST;
    }
  }

  dstOp->main.value = CropAndResize;
}

REGISTER_CONVERTER(CropAndResize, CropAndResize);

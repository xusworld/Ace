//
//  PadTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(PadTf);

tars::OpType PadTf::opType() { return tars::OpType_Padding; }
tars::OpParameter PadTf::type() { return tars::OpParameter_PadParam; }

void PadTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto padparm = new tars::PadParamT;

  padparm->mode = tars::PadValueMode_CONSTANT;
  if (srcNode->opType == "MirrorPad") {
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "mode", value)) {
      if (value.s() == "SYMMETRIC") {
        padparm->mode = tars::PadValueMode_SYMMETRIC;
      } else if (value.s() == "REFLECT") {
        padparm->mode = tars::PadValueMode_REFLECT;
      }
    }
  }

  dstOp->main.value = padparm;
}

REGISTER_CONVERTER(PadTf, Pad);
REGISTER_CONVERTER(PadTf, PadV2);
REGISTER_CONVERTER(PadTf, MirrorPad);

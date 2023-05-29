//
//  GatherV2Tf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(GatherV2);

tars::OpType GatherV2::opType() { return tars::OpType_GatherV2; }
tars::OpParameter GatherV2::type() { return tars::OpParameter_GatherV2; }

void GatherV2::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto GatherV2 = new tars::GatherV2T;
  tensorflow::AttrValue value;

  find_attr_value(srcNode->tfNode, "Taxis", value);
  GatherV2->Taxis = (tars::DataType)value.type();

  find_attr_value(srcNode->tfNode, "Tindices", value);
  GatherV2->Tindices = (tars::DataType)value.type();

  find_attr_value(srcNode->tfNode, "Tparams", value);
  GatherV2->Tparams = (tars::DataType)value.type();

  dstOp->main.value = GatherV2;
}

REGISTER_CONVERTER(GatherV2, GatherV2);

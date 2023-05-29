//
//  PackTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(PackTf);

tars::OpType PackTf::opType() { return tars::OpType_Pack; }
tars::OpParameter PackTf::type() { return tars::OpParameter_PackParam; }

void PackTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto pack = new tars::PackParamT;

  tensorflow::AttrValue value;

  find_attr_value(srcNode->tfNode, "T", value);
  tars::DataType dataType = (tars::DataType)value.type();
  pack->dataType = dataType;

  find_attr_value(srcNode->tfNode, "axis", value);
  pack->axis = value.i();

  dstOp->main.value = pack;
}

REGISTER_CONVERTER(PackTf, Pack);

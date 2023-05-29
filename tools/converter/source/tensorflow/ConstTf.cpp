//
//  ConstTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <map>
#include <string>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"
using namespace tars;
DECLARE_OP_CONVERTER(ConstTf);

tars::OpType ConstTf::opType() { return tars::OpType_Const; }
tars::OpParameter ConstTf::type() { return tars::OpParameter_Blob; }

void ConstTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto parameter = new tars::BlobT;
  tensorflow::AttrValue weightsValue;
  if (!find_attr_value(srcNode->tfNode, "value", weightsValue)) {
    LOG(ERROR) << "Const Node Have Not Data!!!==> " << srcNode->opName;
  }
  tfOpConverter::convertTensorToBlob(parameter, weightsValue.tensor());
  dstOp->main.value = parameter;
  // CHECK(srcNode->inTensors.size() == 0) << "Const Should Not Have Input!!!
  // ===> " << srcNode->opName;
}

REGISTER_CONVERTER(ConstTf, Const);
REGISTER_CONVERTER(ConstTf, HostConst);

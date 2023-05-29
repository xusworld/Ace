//
//  SpaceToDepth.cpp
//  MNNConverter
//
//  Created by MNN on 2019/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(SpaceToDepthTf);

tars::OpType SpaceToDepthTf::opType() { return tars::OpType_SpaceToDepth; }
tars::OpParameter SpaceToDepthTf::type() {
  return tars::OpParameter_DepthSpaceParam;
}

// input: tensor
void SpaceToDepthTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto spaceToDepthParam = new tars::DepthSpaceParamT;
  tensorflow::AttrValue value;

  if (find_attr_value(srcNode->tfNode, "block_size", value)) {
    spaceToDepthParam->blockSize = value.i();
  } else {
    DLOG(ERROR) << "block_size not found";
  }

  dstOp->main.value = spaceToDepthParam;
}

REGISTER_CONVERTER(SpaceToDepthTf, SpaceToDepth);

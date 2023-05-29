//
//  BatchMatMulTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(BatchMatMulTf);

tars::OpType BatchMatMulTf::opType() { return tars::OpType_BatchMatMul; }

tars::OpParameter BatchMatMulTf::type() {
  return tars::OpParameter_BatchMatMulParam;
}

void BatchMatMulTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto batchMatMulParam = new tars::BatchMatMulParamT;

  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "adj_x", value)) {
    batchMatMulParam->adjX = value.b();
  }

  if (find_attr_value(srcNode->tfNode, "adj_y", value)) {
    batchMatMulParam->adjY = value.b();
  }

  dstOp->main.value = batchMatMulParam;
}

REGISTER_CONVERTER(BatchMatMulTf, BatchMatMul);
REGISTER_CONVERTER(BatchMatMulTf, BatchMatMulV2);

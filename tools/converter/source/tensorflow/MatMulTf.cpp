//
//  MatMulTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(MatMulTf);

tars::OpType MatMulTf::opType() { return tars::OpType_MatMul; }
tars::OpParameter MatMulTf::type() { return tars::OpParameter_MatMul; }

void MatMulTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto matmulParam = new tars::MatMulT;

  tensorflow::AttrValue value;

  if (find_attr_value(srcNode->tfNode, "T", value)) {
    matmulParam->T = (tars::DataType)value.type();
  }

  if (find_attr_value(srcNode->tfNode, "transpose_a", value)) {
    matmulParam->transposeA = value.b();
  }

  if (find_attr_value(srcNode->tfNode, "transpose_b", value)) {
    matmulParam->transposeB = value.b();
  }

  dstOp->main.value = matmulParam;
}

REGISTER_CONVERTER(MatMulTf, MatMul);

DECLARE_OP_CONVERTER(MatBandPartTf);

tars::OpType MatBandPartTf::opType() { return tars::OpType_MatrixBandPart; }
tars::OpParameter MatBandPartTf::type() { return tars::OpParameter_NONE; }
void MatBandPartTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  // Do nothing
}

REGISTER_CONVERTER(MatBandPartTf, MatrixBandPart);
REGISTER_CONVERTER(MatBandPartTf, BatchMatrixBandPart);

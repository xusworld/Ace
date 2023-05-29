//
//  ExpandDims.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(ExpandDimsTf);

tars::OpType ExpandDimsTf::opType() { return tars::OpType_ExpandDims; }
tars::OpParameter ExpandDimsTf::type() { return tars::OpParameter_ExpandDims; }

void ExpandDimsTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto parameter = new tars::ExpandDimsT;
#ifdef TF_CONVERT_ORIGIN
  TmpNode *dimNode = tempGraph->_getTmpNode(srcNode->inEdges[1]);

  tensorflow::AttrValue value;
  if (find_attr_value(dimNode->tfNode, "value", value)) {
    const tensorflow::TensorProto &dimTensor = value.tensor();
    parameter->axis = dimTensor.int_val(0);
  }
#endif
  dstOp->main.value = parameter;
}

REGISTER_CONVERTER(ExpandDimsTf, ExpandDims);

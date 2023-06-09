//
//  ReshapeTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(ReshapeTf);
tars::OpType ReshapeTf::opType() { return tars::OpType_Reshape; }
tars::OpParameter ReshapeTf::type() { return tars::OpParameter_Reshape; }

void ReshapeTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto reshape = new tars::ReshapeT;
  dstOp->main.value = reshape;
#ifdef TF_CONVERT_ORIGIN
  TmpNode *shapeNode = tempGraph->_getTmpNode(srcNode->inEdges[1]);
  if (shapeNode->opType != "Const") {
    return;
  }

  // Const Shape
  tensorflow::AttrValue value;
  if (find_attr_value(shapeNode->tfNode, "value", value)) {
    tars::DataType dataType = (tars::DataType)value.tensor().dtype();
    CHECK(dataType == tars::DataType_DT_INT32)
        << "Shape Dtype ERROR" << srcNode->opName;

    reshape->dimType = tars::MNN_DATA_FORMAT_NHWC;

    const int repeatedSize = value.tensor().int_val_size();
    // firstly get value from repeated field
    if (repeatedSize != 0) {
      reshape->dims.resize(repeatedSize);
      for (int i = 0; i < repeatedSize; ++i) {
        reshape->dims[i] = value.tensor().int_val(i);
      }
    } else if (!value.tensor().tensor_content().empty())  // int32
    {
      const int *data = reinterpret_cast<const int *>(
          value.tensor().tensor_content().c_str());
      int size = value.tensor().tensor_content().size() / sizeof(int);
      CHECK(size > 1) << "Shape Data ERROR!!! ===> " << srcNode->opName;
      reshape->dims.resize(size);
      for (int i = 0; i < size; ++i) {
        reshape->dims[i] = data[i];
      }
    } else {
      // only one int value
      reshape->dims.resize(1);
      reshape->dims[0] = value.tensor().int_val(0);
    }
  }
#endif
}

REGISTER_CONVERTER(ReshapeTf, Reshape);

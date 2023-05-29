//
//  DequantizeTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(DequantizeTf);

tars::OpType DequantizeTf::opType() { return tars::OpType_Dequantize; }
tars::OpParameter DequantizeTf::type() { return tars::OpParameter_Dequantize; }

void DequantizeTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto Dequantize = new tars::DequantizeT;
  tensorflow::AttrValue value;

  if (find_attr_value(srcNode->tfNode, "mode", value)) {
    if (value.s() == "MIN_COMBINED") {
      Dequantize->mode = tars::QuantizeMode_MIN_COMBINED;
    } else if (value.s() == "MIN_FIRST") {
      Dequantize->mode = tars::QuantizeMode_MIN_FIRST;
    } else if (value.s() == "SCALED") {
      Dequantize->mode = tars::QuantizeMode_SCALED;
    }
  }

  if (find_attr_value(srcNode->tfNode, "T", value)) {
    const auto dateType = static_cast<tars::DataType>(value.type());
    switch (dateType) {
      case tars::DataType_DT_QUINT8:
        Dequantize->type = tars::DataType_DT_QUINT8;
        break;
      case tars::DataType_DT_QINT8:
        Dequantize->type = tars::DataType_DT_QINT8;
        break;
      case tars::DataType_DT_QUINT16:
        Dequantize->type = tars::DataType_DT_QINT16;
        break;
      case tars::DataType_DT_QINT16:
        Dequantize->type = tars::DataType_DT_QUINT16;
        break;
      case tars::DataType_DT_QINT32:
        Dequantize->type = tars::DataType_DT_QINT32;
        break;
      default:
        DLOG(FATAL) << "unsupported type";
    }
  }

  dstOp->main.value = Dequantize;
}

REGISTER_CONVERTER(DequantizeTf, Dequantize);

//
//  BinaryOpTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(BinartOpTf);

tars::OpType BinartOpTf::opType() { return tars::OpType_BinaryOp; }
tars::OpParameter BinartOpTf::type() { return tars::OpParameter_BinaryOp; }

void BinartOpTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto parameter = new tars::BinaryOpT;

  if (srcNode->opType == "Mul" || srcNode->opType == "LogicalAnd") {
    parameter->opType = tars::BinaryOpOperation_MUL;
  } else if (srcNode->opType == "Sub") {
    parameter->opType = tars::BinaryOpOperation_SUB;
  } else if (srcNode->opType == "Add" || srcNode->opType == "BiasAdd") {
    parameter->opType = tars::BinaryOpOperation_ADD;
  } else if (srcNode->opType == "RealDiv") {
    parameter->opType = tars::BinaryOpOperation_REALDIV;
  } else if (srcNode->opType == "Maximum") {
    parameter->opType = tars::BinaryOpOperation_MAXIMUM;
  } else if (srcNode->opType == "Minimum") {
    parameter->opType = tars::BinaryOpOperation_MINIMUM;
  } else if (srcNode->opType == "Less") {
    parameter->opType = tars::BinaryOpOperation_LESS;
  } else if (srcNode->opType == "LessEqual") {
    parameter->opType = tars::BinaryOpOperation_LESS_EQUAL;
  } else if (srcNode->opType == "GreaterEqual") {
    parameter->opType = tars::BinaryOpOperation_GREATER_EQUAL;
  } else if (srcNode->opType == "Greater") {
    parameter->opType = tars::BinaryOpOperation_GREATER;
  } else if (srcNode->opType == "Equal") {
    parameter->opType = tars::BinaryOpOperation_EQUAL;
  } else if (srcNode->opType == "FloorDiv") {
    parameter->opType = tars::BinaryOpOperation_FLOORDIV;
  } else if (srcNode->opType == "FloorMod") {
    parameter->opType = tars::BinaryOpOperation_FLOORMOD;
  } else if (srcNode->opType == "SquaredDifference") {
    parameter->opType = tars::BinaryOpOperation_SquaredDifference;
  } else if (srcNode->opType == "Pow") {
    parameter->opType = tars::BinaryOpOperation_POW;
  } else if (srcNode->opType == "AddV2") {
    parameter->opType = tars::BinaryOpOperation_ADD;
  } else if (srcNode->opType == "Atan2") {
    parameter->opType = tars::BinaryOpOperation_ATAN2;
  } else if (srcNode->opType == "LogicalOr") {
    parameter->opType = tars::BinaryOpOperation_LOGICALOR;
  } else if (srcNode->opType == "NotEqual") {
    parameter->opType = tars::BinaryOpOperation_NOTEQUAL;
  } else if (srcNode->opType == "TruncateDiv") {
    parameter->opType = tars::BinaryOpOperation_REALDIV;
  } else if (srcNode->opType == "Mod") {
    parameter->opType = tars::BinaryOpOperation_MOD;
  } else {
    DLOG(ERROR) << "MNN Converter Not "
                   "Supported!!!";
  }

  tensorflow::AttrValue value;
  find_attr_value(srcNode->tfNode, "T", value);
  parameter->T = (tars::DataType)value.type();

  dstOp->main.value = parameter;
}

REGISTER_CONVERTER(BinartOpTf, Mul);
REGISTER_CONVERTER(BinartOpTf, LogicalAnd);
REGISTER_CONVERTER(BinartOpTf, Sub);
REGISTER_CONVERTER(BinartOpTf, Add);
REGISTER_CONVERTER(BinartOpTf, Maximum);
REGISTER_CONVERTER(BinartOpTf, RealDiv);
REGISTER_CONVERTER(BinartOpTf, Minimum);
REGISTER_CONVERTER(BinartOpTf, Greater);
REGISTER_CONVERTER(BinartOpTf, Equal);
REGISTER_CONVERTER(BinartOpTf, BiasAdd);
REGISTER_CONVERTER(BinartOpTf, Less);
REGISTER_CONVERTER(BinartOpTf, LessEqual);
REGISTER_CONVERTER(BinartOpTf, GreaterEqual);
REGISTER_CONVERTER(BinartOpTf, FloorDiv);
REGISTER_CONVERTER(BinartOpTf, FloorMod);
REGISTER_CONVERTER(BinartOpTf, SquaredDifference);
REGISTER_CONVERTER(BinartOpTf, Pow);
REGISTER_CONVERTER(BinartOpTf, AddV2);
REGISTER_CONVERTER(BinartOpTf, Atan2);
REGISTER_CONVERTER(BinartOpTf, LogicalOr);
REGISTER_CONVERTER(BinartOpTf, NotEqual);
REGISTER_CONVERTER(BinartOpTf, TruncateDiv);
REGISTER_CONVERTER(BinartOpTf, Mod);

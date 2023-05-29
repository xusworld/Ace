//
//  TensorArrayTf.cpp
//  MNNConverter
//
//  Created by MNN on 2020/12/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

// ============================ TensorArray ============================
DECLARE_OP_CONVERTER(TensorArrayTf);

tars::OpType TensorArrayTf::opType() { return tars::OpType_TensorArray; }
tars::OpParameter TensorArrayTf::type() {
  return tars::OpParameter_TensorArray;
}

void TensorArrayTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto tensorArray = new tars::TensorArrayT;
  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "dtype", value)) {
    tensorArray->T = (tars::DataType)value.type();
  }
  if (find_attr_value(srcNode->tfNode, "dynamic_size", value)) {
    tensorArray->dynamic_size = value.b();
  }
  if (find_attr_value(srcNode->tfNode, "identical_element_shapes", value)) {
    tensorArray->identical_element_shapes = value.b();
  }
  if (find_attr_value(srcNode->tfNode, "element_shape", value)) {
    if (value.shape().dim_size() > 0) {
      tensorArray->element_shape.resize(value.shape().dim_size());
      for (int i = 0; i < value.shape().dim_size(); i++) {
        tensorArray->element_shape[i] = value.shape().dim(i).size();
      }
    }
  }
  dstOp->main.value = tensorArray;
}

REGISTER_CONVERTER(TensorArrayTf, TensorArrayV3);

// ============================ TensorArraySize ============================
DECLARE_OP_CONVERTER(TensorArraySizeTf);

tars::OpType TensorArraySizeTf::opType() {
  return tars::OpType_TensorArraySize;
}
tars::OpParameter TensorArraySizeTf::type() { return tars::OpParameter_NONE; }

void TensorArraySizeTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(TensorArraySizeTf, TensorArraySizeV3);

// ============================ TensorArrayRead ============================
DECLARE_OP_CONVERTER(TensorArrayReadTf);

tars::OpType TensorArrayReadTf::opType() {
  return tars::OpType_TensorArrayRead;
}
tars::OpParameter TensorArrayReadTf::type() {
  return tars::OpParameter_TensorArray;
}

void TensorArrayReadTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto tensorArrayRead = new tars::TensorArrayT;
  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "dtype", value)) {
    tensorArrayRead->T = (tars::DataType)value.type();
  }
  dstOp->main.value = tensorArrayRead;
}

REGISTER_CONVERTER(TensorArrayReadTf, TensorArrayReadV3);

// ============================ TensorArrayWrite ============================
DECLARE_OP_CONVERTER(TensorArrayWriteTf);

tars::OpType TensorArrayWriteTf::opType() {
  return tars::OpType_TensorArrayWrite;
}
tars::OpParameter TensorArrayWriteTf::type() {
  return tars::OpParameter_TensorArray;
}

void TensorArrayWriteTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto tensorArrayWrite = new tars::TensorArrayT;
  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "T", value)) {
    tensorArrayWrite->T = (tars::DataType)value.type();
  }
  dstOp->main.value = tensorArrayWrite;
}

REGISTER_CONVERTER(TensorArrayWriteTf, TensorArrayWriteV3);

// ============================ TensorArrayGather ============================
DECLARE_OP_CONVERTER(TensorArrayGatherTf);

tars::OpType TensorArrayGatherTf::opType() {
  return tars::OpType_TensorArrayGather;
}
tars::OpParameter TensorArrayGatherTf::type() {
  return tars::OpParameter_TensorArray;
}

void TensorArrayGatherTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto tensorArrayGather = new tars::TensorArrayT;
  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "dtype", value)) {
    tensorArrayGather->T = (tars::DataType)value.type();
  }
  if (find_attr_value(srcNode->tfNode, "element_shape", value)) {
    if (value.shape().dim_size() > 0) {
      tensorArrayGather->element_shape.resize(value.shape().dim_size());
      for (int i = 0; i < value.shape().dim_size(); i++) {
        tensorArrayGather->element_shape[i] = value.shape().dim(i).size();
      }
    }
  }
  dstOp->main.value = tensorArrayGather;
}

REGISTER_CONVERTER(TensorArrayGatherTf, TensorArrayGatherV3);

// ============================ TensorArrayScatter ============================
DECLARE_OP_CONVERTER(TensorArrayScatterTf);

tars::OpType TensorArrayScatterTf::opType() {
  return tars::OpType_TensorArrayScatter;
}
tars::OpParameter TensorArrayScatterTf::type() {
  return tars::OpParameter_TensorArray;
}

void TensorArrayScatterTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto tensorArrayScatter = new tars::TensorArrayT;
  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "T", value)) {
    tensorArrayScatter->T = (tars::DataType)value.type();
  }
  dstOp->main.value = tensorArrayScatter;
}

REGISTER_CONVERTER(TensorArrayScatterTf, TensorArrayScatterV3);

// ============================ TensorArraySplit ============================
DECLARE_OP_CONVERTER(TensorArraySplitTf);

tars::OpType TensorArraySplitTf::opType() {
  return tars::OpType_TensorArraySplit;
}
tars::OpParameter TensorArraySplitTf::type() {
  return tars::OpParameter_TensorArray;
}

void TensorArraySplitTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto tensorArraySplit = new tars::TensorArrayT;
  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "T", value)) {
    tensorArraySplit->T = (tars::DataType)value.type();
  }
  dstOp->main.value = tensorArraySplit;
}

REGISTER_CONVERTER(TensorArraySplitTf, TensorArraySplitV3);

// ============================ TensorArrayConcat ============================
DECLARE_OP_CONVERTER(TensorArrayConcatTf);

tars::OpType TensorArrayConcatTf::opType() {
  return tars::OpType_TensorArrayConcat;
}
tars::OpParameter TensorArrayConcatTf::type() {
  return tars::OpParameter_TensorArray;
}

void TensorArrayConcatTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  auto tensorArrayConcat = new tars::TensorArrayT;
  tensorflow::AttrValue value;
  if (find_attr_value(srcNode->tfNode, "T", value)) {
    tensorArrayConcat->T = (tars::DataType)value.type();
  }
  if (find_attr_value(srcNode->tfNode, "element_shape", value)) {
    if (value.shape().dim_size() > 0) {
      tensorArrayConcat->element_shape.resize(value.shape().dim_size());
      for (int i = 0; i < value.shape().dim_size(); i++) {
        tensorArrayConcat->element_shape[i] = value.shape().dim(i).size();
      }
    }
  }
  dstOp->main.value = tensorArrayConcat;
}

REGISTER_CONVERTER(TensorArrayConcatTf, TensorArrayConcatV3);

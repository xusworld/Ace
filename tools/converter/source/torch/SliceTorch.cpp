//
//  SliceTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(SliceTorch);

tars::OpType SliceTorch::opType() { return tars::OpType_Slice; }
tars::OpParameter SliceTorch::type() { return tars::OpParameter_Slice; }
std::vector<int> SliceTorch::inputTensorIdx() { return {0}; }

void SliceTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                     TorchScope* scope) {
  auto param = new tars::SliceT;
  const std::string opType = getRealOpType(node);
  if (opType == "chunk") {
    param->axis = getValue<int64_t>(node->input(2));
    param->sourceType = tars::NetSource_TENSORFLOW;
  } else if (opType == "ListUnpack") {
    param->axis = 0;
    param->sourceType = tars::NetSource_TENSORFLOW;
  }
  dstOp->main.value = param;
}

REGISTER_CONVERTER(SliceTorch, chunk);
REGISTER_CONVERTER(SliceTorch, ListUnpack);

DECLARE_OP_CONVERTER(StridedSliceTorch);

tars::OpType StridedSliceTorch::opType() { return tars::OpType_Extra; }
tars::OpParameter StridedSliceTorch::type() { return tars::OpParameter_Extra; }
std::vector<int> StridedSliceTorch::inputTensorIdx() { return {-1}; }

void StridedSliceTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                            TorchScope* scope) {
  auto extra = new tars::ExtraT;
  dstOp->main.value = extra;
  extra->engine = "Torch";
  extra->type = getRealOpType(node);
  if (node->inputs().size() > 1) {
    auto dim = node->input(1);
    if (toIValue(dim)) {
      std::unique_ptr<tars::AttributeT> dimAttr(new tars::AttributeT);
      dimAttr->key = "dim";
      dimAttr->i = getValue<int64_t>(dim);
      extra->attr.emplace_back(std::move(dimAttr));
    }
  }
  if (node->inputs().size() > 2) {
    auto start = node->input(2);
    if (toIValue(start)) {
      std::unique_ptr<tars::AttributeT> startAttr(new tars::AttributeT);
      startAttr->key = "start";
      startAttr->i = getValue<int64_t>(start);
      extra->attr.emplace_back(std::move(startAttr));
    }
  }
  if (node->inputs().size() > 3) {
    auto end = node->input(3);
    if (toIValue(end)) {
      std::unique_ptr<tars::AttributeT> endAttr(new tars::AttributeT);
      endAttr->key = "end";
      endAttr->i =
          std::min(getValue<int64_t>(end),
                   static_cast<int64_t>(std::numeric_limits<int>::max()));
      extra->attr.emplace_back(std::move(endAttr));
    }
  }
  if (node->inputs().size() > 4) {
    auto stride = node->input(4);
    if (toIValue(stride)) {
      std::unique_ptr<tars::AttributeT> strideAttr(new tars::AttributeT);
      strideAttr->key = "stride";
      strideAttr->i = getValue<int64_t>(stride);
      extra->attr.emplace_back(std::move(strideAttr));
    }
  } else {
    std::unique_ptr<tars::AttributeT> strideAttr(new tars::AttributeT);
    strideAttr->key = "stride";
    strideAttr->i = 1;
    extra->attr.emplace_back(std::move(strideAttr));
  }
}

REGISTER_CONVERTER(StridedSliceTorch, slice);

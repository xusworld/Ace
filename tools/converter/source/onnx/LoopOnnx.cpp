//
//  LoopOnnx.cpp
//  MNN
//
//  Created by MNN on 2021/06/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include <stdio.h>

#include "onnxOpConverter.hpp"
DECLARE_OP_CONVERTER(LoopOnnx);

tars::OpType LoopOnnx::opType() { return tars::OpType_While; }
tars::OpParameter LoopOnnx::type() { return tars::OpParameter_WhileParam; }

void LoopOnnx::run(tars::OpT* dstOp, const onnx::NodeProto* onnxNode,
                   OnnxScope* scope) {
  if (onnxNode->input(0) == "" || onnxNode->input(1) == "") {
    MNN_ERROR("Failed: Loop don't support optional M and cond input\n");
    return;
  }
  auto param = new tars::WhileParamT;
  dstOp->name += "/Loop";
  param->body_graph = dstOp->name + "/body";
  auto body = &onnxNode->attribute(0).g();
  // build body
  std::string empty;
  int N = onnxNode->input_size() - 2;
  int K = onnxNode->output_size() - N;
  MNN_ASSERT(body->input_size() == N + 2);
  MNN_ASSERT(body->output_size() == N + K + 1);
  auto ousideInputs = scope->buildSubGraph(body, param->body_graph, true);
  std::vector<int> outsideIndexOutside(ousideInputs.size());
  for (int i = 0; i < ousideInputs.size(); ++i) {
    // subgraph own by LOOP may introduce extra input which is not exist on
    // current graph, create corresponding input op here

    scope->addInputForOp(dstOp, ousideInputs[i], true);
    outsideIndexOutside[i] =
        scope->declareTensor(dstOp->name + "_extra_unused_" + ousideInputs[i]);
  }
  dstOp->outputIndexes.insert(dstOp->outputIndexes.begin() + N,
                              outsideIndexOutside.begin(),
                              outsideIndexOutside.end());
  // update i
  dstOp->main.value = param;
}

REGISTER_CONVERTER(LoopOnnx, Loop);

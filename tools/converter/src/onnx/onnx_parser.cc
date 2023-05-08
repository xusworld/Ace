#include <flatbuffers/idl.h>
#include <flatbuffers/minireflect.h>
#include <flatbuffers/util.h>
#include <glog/logging.h>

#include <iostream>

#include "ace_generated.h"
#include "onnx.pb.h"
#include "onnxConverter.hpp"
#include "onnx_node_parser_manager.h"
#include "onnx_tmp_graph.h"
#include "utils.h"

namespace ace {
namespace parser {

int OnnxToAceModel(const std::string& onnx_model_path,
                   const std::string& bizCode,
                   std::unique_ptr<ace::NetT>& netT) {
  // Set deep learning framework
  netT->sourceType = ace::FrontendFramework_ONNX;
  netT->bizCode = bizCode;

  std::map<std::string, ace::OpT*> nodesMap;
  std::map<std::string, int> name2InputIdx;

  onnx::ModelProto onnx_model;

  bool success = OnnxReadProtoFromBinary(onnx_model_path, &onnx_model);
  CHECK(success) << "Read onnx model failed: " << onnx_model_path;
  LOG(INFO) << "ONNX IR version: " << onnx_model.ir_version();
  LOG(INFO) << "ONNX Model version:" << onnx_model.model_version();

  // onnx compute graph
  const auto& onnx_graph = onnx_model.graph();
  const int nodeCount = onnx_graph.node_size();
  std::shared_ptr<OnnxTmpGraph> onnxTmpGraph(new OnnxTmpGraph(&onnx_graph));

  // auto onnxTmpGraph = BuildOnnxGraph(onnx_model_path);
  const auto& initializers = onnxTmpGraph->GetModelInitializers();
  const auto& inputs = onnxTmpGraph->GetModelInputs();
  const auto& outputs = onnxTmpGraph->GetModelOutputs();

  // Handle inputs
  LOG(INFO) << "inputs.size(): " << inputs.size();
  for (const auto& input : inputs) {
    if (initializers.find(input.first) == initializers.end()) {
      netT->tensorName.push_back(input.first);
      name2InputIdx.insert(std::make_pair(input.first, name2InputIdx.size()));
    }
  }

  for (const auto& iter : name2InputIdx) {
    ace::OpT* op = new ace::OpT;
    op->name = iter.first;
    op->type = ace::OpType_Input;
    op->main.type = ace::OpParameter_Input;

    auto inputParam = new ace::InputT;
    const auto it = inputs.find(iter.first);
    DCHECK(it != inputs.end()) << "Input Paramter ERROR ==> " << iter.first;

    auto info = it->second;
    const auto& tensorInfo = (it->second)->type().tensor_type();
    const int inputDimSize = tensorInfo.shape().dim_size();
    inputParam->dims.resize(inputDimSize);
    for (int i = 0; i < inputDimSize; ++i) {
      inputParam->dims[i] = tensorInfo.shape().dim(i).dim_value();
    }

    inputParam->dtype = ToAceDataType(tensorInfo.elem_type());
    inputParam->dformat = ace::DATA_FORMAT_NCHW;
    op->outputIndexes.push_back(name2InputIdx[iter.first]);
    op->main.value = inputParam;
    nodesMap.insert(std::make_pair(iter.first, op));
    netT->oplists.emplace_back(op);
  }

  for (int i = 0; i < onnxTmpGraph->GetOnnxGraph()->node_size(); ++i) {
    const auto& onnx_node = onnxTmpGraph->GetOnnxGraph()->node(i);
    const auto& op_type = onnx_node.op_type();

    auto opConverter = OnnxNodeParserManager::Global()->Get(op_type);

    // create a new ace operator
    ace::OpT* op = new ace::OpT;
    op->name = onnx_node.output(0);
    op->type = opConverter->opType();
    op->main.type = opConverter->type();
    nodesMap.insert(std::make_pair(onnx_node.output(0), op));

    // convert initializer to be Constant node(op)
    for (int k = 0; k < onnx_node.input_size(); ++k) {
      const auto& inputName = onnx_node.input(k);
      const auto it = initializers.find(inputName);
      if (it != initializers.end() &&
          name2InputIdx.find(it->first) == name2InputIdx.end()) {
        // Create const Op
        ace::OpT* constOp = new ace::OpT;
        constOp->type = ace::OpType_Const;
        constOp->main.type = ace::OpParameter_Blob;
        constOp->main.value = OnnxTensorToBlob(it->second);
        nodesMap.insert(std::make_pair(inputName, constOp));
        auto outputIndex = (int)netT->tensorName.size();
        constOp->name = it->first;
        constOp->outputIndexes.push_back(outputIndex);
        name2InputIdx.insert(std::make_pair(it->first, outputIndex));
        netT->tensorName.emplace_back(constOp->name);
        netT->oplists.emplace_back(constOp);
      }
    }

    // TODO, delete the run() args opInitializers
    std::vector<const onnx::TensorProto*> opInitializers;
    for (int k = 0; k < onnx_node.input_size(); ++k) {
      const auto& inputName = onnx_node.input(k);
      const auto it = initializers.find(inputName);
      if (it != initializers.end()) {
        opInitializers.push_back(it->second);
      }
    }
    opConverter->parse(op, &onnx_node, opInitializers);

    netT->oplists.emplace_back(op);

    const int outputTensorSize = onnx_node.output_size();
    for (int ot = 0; ot < outputTensorSize; ++ot) {
      netT->tensorName.push_back(onnx_node.output(ot));
      name2InputIdx.insert(
          std::make_pair(onnx_node.output(ot), name2InputIdx.size()));
    }
  }

  // set input-output tensor's index
  for (int i = 0; i < onnxTmpGraph->GetOnnxGraph()->node_size(); ++i) {
    const auto& onnxNode = onnxTmpGraph->GetOnnxGraph()->node(i);

    auto iter = nodesMap.find(onnxNode.output(0));
    DCHECK(iter != nodesMap.end()) << "Can't find node: " << onnxNode.name();
    auto curOp = nodesMap[onnxNode.output(0)];

    // set input index
    const int inputSize = onnxNode.input_size();
    for (int j = 0; j < inputSize; ++j) {
      const auto& inputName = onnxNode.input(j);
      // onnx have optional input, which may be a placeholder when pytorch
      // export onnx model, so drop this input, but we should check it out
      // sometimes.
      if (inputName == "") {
        LOG(INFO) << "Check it out ==> " << curOp->name
                  << " has empty input, the index is " << j;
        continue;
      }
      auto iterTensor = name2InputIdx.find(inputName);
      DCHECK(iterTensor != name2InputIdx.end())
          << "Can't find tensor: " << inputName;
      curOp->inputIndexes.push_back(iterTensor->second);
    }

    // set output index
    const int outputSize = onnxNode.output_size();
    for (int j = 0; j < outputSize; ++j) {
      const auto& outputName = onnxNode.output(j);
      auto iterTensor = name2InputIdx.find(outputName);
      DCHECK(iterTensor != name2InputIdx.end())
          << "Can't find tensor: " << outputName;
      curOp->outputIndexes.push_back(iterTensor->second);
    }
  }

  netT->tensorNumber = name2InputIdx.size();

  for (const auto& iter : outputs) {
    netT->outputName.push_back(iter.first);
  }

  return 0;
}

}  // namespace parser
}  // namespace ace
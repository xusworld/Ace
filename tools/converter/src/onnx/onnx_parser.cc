#include <flatbuffers/idl.h>
#include <flatbuffers/minireflect.h>
#include <flatbuffers/util.h>

#include <iostream>

#include "ace_generated.h"
#include "logkit.h"
#include "onnx.pb.h"
#include "onnxConverter.hpp"
#include "onnx_node_parser_manager.h"
#include "onnx_tmp_graph.h"
#include "utils.h"

namespace ace {
namespace parser {

int OnnxToAceModel(const std::string inputModel, const std::string bizCode,
                   std::unique_ptr<ace::NetT>& netT) {
  onnx::ModelProto onnxModel;
  // read ONNX Model
  bool success = ace::parser::OnnxReadProtoFromBinary(inputModel, &onnxModel);
  DCHECK(success) << "read onnx model failed: " << inputModel;

  LOG(INFO) << "ONNX Model ir version: " << onnxModel.ir_version();

  const auto& onnxGraph = onnxModel.graph();
  const int nodeCount = onnxGraph.node_size();

  std::shared_ptr<OnnxTmpGraph> onnxTempGraph(new OnnxTmpGraph(&onnxGraph));

  // op_name: name
  std::map<std::string, ace::OpT*> mnnNodesMap;
  // all tensors container
  std::map<std::string, int> tensorsName;
  // find the inputs which do not have initializer
  const auto& initializers = onnxTempGraph->mInitializers;
  const auto& inputs = onnxTempGraph->mInputs;
  const auto& outputs = onnxTempGraph->mOutputs;
  const auto& constantNodeToDelete = onnxTempGraph->mConstantNodeToDelete;
  for (const auto& iter : inputs) {
    bool notHaveInitializer =
        initializers.find(iter.first) == initializers.end();
    if (notHaveInitializer) {
      netT->tensorName.push_back(iter.first);
      tensorsName.insert(std::make_pair(iter.first, tensorsName.size()));
    }
  }

  // set input node to MNN net
  for (const auto& iter : tensorsName) {
    // here tensorsName are true Input node name
    ace::OpT* op = new ace::OpT;
    op->name = iter.first;
    op->type = ace::OpType_Input;
    op->main.type = ace::OpParameter_Input;
    auto inputParam = new ace::InputT;
    const auto it = inputs.find(iter.first);
    DCHECK(it != inputs.end()) << "Input Paramter ERROR ==> " << iter.first;
    const auto& tensorInfo = (it->second)->type().tensor_type();
    const int inputDimSize = tensorInfo.shape().dim_size();
    inputParam->dims.resize(inputDimSize);
    for (int i = 0; i < inputDimSize; ++i) {
      inputParam->dims[i] = tensorInfo.shape().dim(i).dim_value();
    }
    inputParam->dtype = ToAceDataType(tensorInfo.elem_type());
    inputParam->dformat = ace::DATA_FORMAT_NCHW;
    op->outputIndexes.push_back(tensorsName[iter.first]);
    op->main.value = inputParam;
    mnnNodesMap.insert(std::make_pair(iter.first, op));
    netT->oplists.emplace_back(op);
  }

  for (int i = 0; i < nodeCount; ++i) {
    const auto& onnxNode = onnxGraph.node(i);
    const auto& opType = onnxNode.op_type();

    // name maybe null, use the first output name as node-name
    const auto& name = onnxNode.output(0);

    // TODO not to use constantNodeToDelete anymore
    if (constantNodeToDelete.find(name) != constantNodeToDelete.end()) {
      continue;
    }

    LOG(INFO) << "OpType: " << opType;
    auto opConverter = OnnxNodeParserManager::Global()->Get(opType);

    ace::OpT* op = new ace::OpT;
    op->name = name;
    op->type = opConverter->opType();
    op->main.type = opConverter->type();
    mnnNodesMap.insert(std::make_pair(name, op));

    // convert initializer to be Constant node(op)
    for (int k = 0; k < onnxNode.input_size(); ++k) {
      const auto& inputName = onnxNode.input(k);
      const auto it = initializers.find(inputName);
      if (it != initializers.end() &&
          tensorsName.find(it->first) == tensorsName.end()) {
        // Create const Op
        ace::OpT* constOp = new ace::OpT;
        constOp->type = ace::OpType_Const;
        constOp->main.type = ace::OpParameter_Blob;
        constOp->main.value = OnnxTensorToBlob(it->second);
        mnnNodesMap.insert(std::make_pair(inputName, constOp));
        auto outputIndex = (int)netT->tensorName.size();
        constOp->name = it->first;
        constOp->outputIndexes.push_back(outputIndex);
        tensorsName.insert(std::make_pair(it->first, outputIndex));
        netT->tensorName.emplace_back(constOp->name);
        netT->oplists.emplace_back(constOp);
      }
    }

    // TODO, delete the run() args opInitializers
    std::vector<const onnx::TensorProto*> opInitializers;
    for (int k = 0; k < onnxNode.input_size(); ++k) {
      const auto& inputName = onnxNode.input(k);
      const auto it = initializers.find(inputName);
      if (it != initializers.end()) {
        opInitializers.push_back(it->second);
      }
    }
    opConverter->parse(op, &onnxNode, opInitializers);

    netT->oplists.emplace_back(op);

    const int outputTensorSize = onnxNode.output_size();
    for (int ot = 0; ot < outputTensorSize; ++ot) {
      netT->tensorName.push_back(onnxNode.output(ot));
      tensorsName.insert(
          std::make_pair(onnxNode.output(ot), tensorsName.size()));
    }
  }

  // set input-output tensor's index
  for (int i = 0; i < nodeCount; ++i) {
    const auto& onnxNode = onnxGraph.node(i);

    auto iter = mnnNodesMap.find(onnxNode.output(0));
    DCHECK(iter != mnnNodesMap.end()) << "Can't find node: " << onnxNode.name();
    auto curOp = mnnNodesMap[onnxNode.output(0)];

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
      auto iterTensor = tensorsName.find(inputName);
      DCHECK(iterTensor != tensorsName.end())
          << "Can't find tensor: " << inputName;
      curOp->inputIndexes.push_back(iterTensor->second);
    }

    // set output index
    const int outputSize = onnxNode.output_size();
    for (int j = 0; j < outputSize; ++j) {
      const auto& outputName = onnxNode.output(j);
      auto iterTensor = tensorsName.find(outputName);
      DCHECK(iterTensor != tensorsName.end())
          << "Can't find tensor: " << outputName;
      curOp->outputIndexes.push_back(iterTensor->second);
    }
  }

  netT->tensorNumber = tensorsName.size();
  // set MNN net output name
  for (const auto& iter : outputs) {
    netT->outputName.push_back(iter.first);
  }

  netT->sourceType = ace::NetSource_ONNX;
  netT->bizCode = bizCode;

  return 0;
}

}  // namespace parser
}  // namespace ace
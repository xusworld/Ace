#pragma once

#include <stdio.h>

#include "onnx.pb.h"

class OnnxTmpNode final {
 public:
  OnnxTmpNode() = default;
  ~OnnxTmpNode() = default;

  const onnx::NodeProto* node;
  std::string op_name;
  std::string op_type;

  std::set<std::string> in_edges;
  std::set<std::string> out_edges;
};

class OnnxTmpGraph final {
 public:
  OnnxTmpGraph(const onnx::GraphProto* onnxGraph);
  OnnxTmpGraph() = delete;
  ~OnnxTmpGraph() = default;

  int buildGraph();
  std::shared_ptr<OnnxTmpNode> getTmpNode(const std::string& nodeName);
  std::map<std::string, const onnx::TensorProto*> GetModelInitializers();
  std::map<std::string, const onnx::ValueInfoProto*> GetModelInputs();
  std::map<std::string, const onnx::ValueInfoProto*> GetModelOutputs();
  std::set<std::string> GetNodesType();
  const onnx::GraphProto* GetOnnxGraph() { return graph_; };

 private:
  void init();
  void genMinGraph();
  int makeConnection(const std::shared_ptr<OnnxTmpNode>& srcNode,
                     const std::shared_ptr<OnnxTmpNode>& dstNode,
                     const std::string& srcName, const std::string& dstName);

  const onnx::GraphProto* graph_;
  // The nodes in the graph, sorted topologically.
  std::map<std::string, std::shared_ptr<OnnxTmpNode>> nodes_;
  //  The name of the graph.
  std::string name_;
  // A list of named tensor values, used to specify constant inputs of the
  // graph.Each TensorProto entry must have a distinct name (within the list)
  // that MAY also appear in the input list.
  std::map<std::string, const onnx::TensorProto*> initializers_;
  // The inputs and outputs of the graph.
  std::map<std::string, const onnx::ValueInfoProto*> inputs_;
  std::map<std::string, const onnx::ValueInfoProto*> outputs_;
  std::set<std::string> nodes_type_;
};

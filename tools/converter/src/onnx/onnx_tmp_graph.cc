#include <glog/logging.h>

#include "onnx.pb.h"
#include "onnx_tmp_graph.h"

OnnxTmpGraph::OnnxTmpGraph(const onnx::GraphProto* graph) : graph_(graph) {
  init();
}

void OnnxTmpGraph::init() {
  // model info
  LOG(INFO) << "Nodes : " << graph_->node_size();
  LOG(INFO) << "Initializers: " << graph_->initializer_size();
  LOG(INFO) << "Inputs : " << graph_->input_size();
  LOG(INFO) << "Outputs : " << graph_->output_size();

  // nodes
  for (int i = 0; i < graph_->node_size(); ++i) {
    auto& onnx_node = graph_->node(i);
    std::shared_ptr<OnnxTmpNode> node(new OnnxTmpNode());
    const auto name = onnx_node.name();
    node->op_name = onnx_node.output(0);
    node->op_type = onnx_node.op_type();
    node->node = &onnx_node;

    LOG(INFO) << "Name: " << name << " OpName: " << node->op_name
              << " OpType: " << node->op_type;

    nodes_type_.insert(node->op_type);

    nodes_.insert(std::make_pair(onnx_node.output(0), node));
  }
  // names
  name_ = graph_->name();

  // initializers
  for (int i = 0; i < graph_->initializer_size(); ++i) {
    const auto& initializer = graph_->initializer(i);
    initializers_.insert(std::make_pair(initializer.name(), &initializer));
  }

  // inputs & outputs
  for (int i = 0; i < graph_->input_size(); ++i) {
    const auto& input = graph_->input(i);
    inputs_.insert(std::make_pair(input.name(), &input));
  }
  for (int i = 0; i < graph_->output_size(); ++i) {
    const auto& output = graph_->output(i);
    outputs_.insert(std::make_pair(output.name(), &output));
  }

  LOG(INFO) << "OnnxTmpGraph build success";
}

int OnnxTmpGraph::buildGraph() {
  for (int i = 0; i < graph_->node_size(); ++i) {
    const auto& onnx_node = graph_->node(i);
    const std::string nodeName = onnx_node.output(0);
    const auto& curNode = getTmpNode(nodeName);

    // 遍历当前节点的所有输入节点
    for (int j = 0; j < onnx_node.input_size(); ++j) {
      const std::string input_name = onnx_node.input(j);
      const auto& src_node = getTmpNode(input_name);
      if (!src_node) continue;
      makeConnection(src_node, curNode, input_name, nodeName);
    }
  }

  return 0;
}

int OnnxTmpGraph::makeConnection(const std::shared_ptr<OnnxTmpNode>& srcNode,
                                 const std::shared_ptr<OnnxTmpNode>& dstNode,
                                 const std::string& srcName,
                                 const std::string& dstName) {
  // srcNode 代表前序节点，dstNode
  // 代表当前节点。向前序节点的输出中加入当前节点,向当前节点的输入中加入前序节点
  srcNode->out_edges.insert(dstName);
  dstNode->in_edges.insert(srcName);

  return 0;
}

std::shared_ptr<OnnxTmpNode> OnnxTmpGraph::getTmpNode(const std::string& name) {
  const auto& it = nodes_.find(name);
  if (it != nodes_.end()) {
    return it->second;
  }
  LOG(FATAL) << "Can't not find node " << name << " ]";
  return 0;
}

std::map<std::string, const onnx::TensorProto*>
OnnxTmpGraph::GetModelInitializers() {
  return initializers_;
}

std::map<std::string, const onnx::ValueInfoProto*>
OnnxTmpGraph::GetModelInputs() {
  return inputs_;
}

std::map<std::string, const onnx::ValueInfoProto*>
OnnxTmpGraph::GetModelOutputs() {
  return outputs_;
}

std::set<std::string> OnnxTmpGraph::GetNodesType() { return nodes_type_; }

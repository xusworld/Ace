#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(ExpandOnnx);

ace::OpType ExpandOnnx::opType() { return ace::OpType_BroadcastTo; }

ace::OpParameter ExpandOnnx::type() { return ace::OpParameter_NONE; }

void ExpandOnnx::parse(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                       std::vector<const onnx::TensorProto *> initializers) {
  DCHECK(2 == onnxNode->input_size()) << "ONNX Expand should have 2 inputs!";
  return;
}

REGISTER_ONNX_NODE_PARSER(ExpandOnnx, Expand);
}  // namespace parser
}  // namespace ace
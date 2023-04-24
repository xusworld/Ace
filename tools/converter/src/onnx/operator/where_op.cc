#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(WhereOnnx);

ace::OpType WhereOnnx::opType() { return ace::OpType_Select; }

ace::OpParameter WhereOnnx::type() { return ace::OpParameter_NONE; }

void WhereOnnx::parse(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                      std::vector<const onnx::TensorProto *> initializers) {
  return;
}

REGISTER_ONNX_NODE_PARSER(WhereOnnx, Where);
}  // namespace parser
}  // namespace ace

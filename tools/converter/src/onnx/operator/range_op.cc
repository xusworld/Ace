#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(RangeOnnx);

ace::OpType RangeOnnx::opType() { return ace::OpType_Range; }

ace::OpParameter RangeOnnx::type() { return ace::OpParameter_NONE; }

void RangeOnnx::parse(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                      std::vector<const onnx::TensorProto *> initializers) {
  return;
}

REGISTER_ONNX_NODE_PARSER(RangeOnnx, Range);

}  // namespace parser
}  // namespace ace

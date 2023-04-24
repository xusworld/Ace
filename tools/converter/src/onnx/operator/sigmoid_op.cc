#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(SigmoidOnnx);

ace::OpType SigmoidOnnx::opType() { return ace::OpType_Sigmoid; }

ace::OpParameter SigmoidOnnx::type() { return ace::OpParameter_NONE; }

void SigmoidOnnx::parse(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                        std::vector<const onnx::TensorProto *> initializers) {
  return;
}

REGISTER_ONNX_NODE_PARSER(SigmoidOnnx, Sigmoid);
}  // namespace parser
}  // namespace ace

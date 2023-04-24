#include <stdio.h>

#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(TanhOnnx);

ace::OpType TanhOnnx::opType() { return ace::OpType_TanH; }
ace::OpParameter TanhOnnx::type() { return ace::OpParameter_NONE; }

void TanhOnnx::parse(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                     std::vector<const onnx::TensorProto*> initializers) {
  dstOp->main.value = nullptr;
}

REGISTER_ONNX_NODE_PARSER(TanhOnnx, Tanh);
}  // namespace parser
}  // namespace ace
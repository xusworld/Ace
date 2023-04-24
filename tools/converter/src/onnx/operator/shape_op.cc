#include <stdio.h>

#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(ShapeOnnx);

ace::OpType ShapeOnnx::opType() { return ace::OpType_Shape; }
ace::OpParameter ShapeOnnx::type() { return ace::OpParameter_NONE; }

void ShapeOnnx::parse(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      std::vector<const onnx::TensorProto*> initializers) {
  dstOp->defaultDimentionFormat = ace::DATA_FORMAT_NCHW;
}

REGISTER_ONNX_NODE_PARSER(ShapeOnnx, Shape);
}  // namespace parser
}  // namespace ace

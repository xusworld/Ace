#include <stdio.h>

#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(ReshapeOnnx);

ace::OpType ReshapeOnnx::opType() { return ace::OpType_Reshape; }
ace::OpParameter ReshapeOnnx::type() { return ace::OpParameter_Reshape; }

void ReshapeOnnx::parse(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                        std::vector<const onnx::TensorProto*> initializers) {
  auto para = new ace::ReshapeT;
  para->dimType = ace::DATA_FORMAT_NCHW;
  dstOp->main.value = para;
}

REGISTER_ONNX_NODE_PARSER(ReshapeOnnx, Reshape);
}  // namespace parser
}  // namespace ace

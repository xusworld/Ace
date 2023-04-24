#include <stdio.h>

#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(SoftmaxOnnx);

ace::OpType SoftmaxOnnx::opType() { return ace::OpType_Softmax; }
ace::OpParameter SoftmaxOnnx::type() { return ace::OpParameter_Axis; }

void SoftmaxOnnx::parse(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                        std::vector<const onnx::TensorProto*> initializers) {
  auto axis = new ace::AxisT;
  axis->axis = 1;
  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "axis") {
      axis->axis = (int)attributeProto.i();
    }
  }
  dstOp->main.value = axis;
}

REGISTER_ONNX_NODE_PARSER(SoftmaxOnnx, Softmax);
}  // namespace parser
}  // namespace ace

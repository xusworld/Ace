#include <stdio.h>

#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(ConcatOnnx);

ace::OpType ConcatOnnx::opType() { return ace::OpType_Concat; }
ace::OpParameter ConcatOnnx::type() { return ace::OpParameter_Axis; }

void ConcatOnnx::parse(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                       std::vector<const onnx::TensorProto*> initializers) {
  auto para = new ace::AxisT;
  para->axis = 0;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "axis") {
      para->axis = attributeProto.i();
    }
  }

  dstOp->main.value = para;
}

REGISTER_ONNX_NODE_PARSER(ConcatOnnx, Concat);

}  // namespace parser
}  // namespace ace

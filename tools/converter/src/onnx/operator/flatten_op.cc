#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(FlattenOnnx);

ace::OpType FlattenOnnx::opType() { return ace::OpType_Flatten; }

ace::OpParameter FlattenOnnx::type() { return ace::OpParameter_Flatten; }

void FlattenOnnx::parse(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                        std::vector<const onnx::TensorProto *> initializers) {
  auto param = new ace::FlattenT;

  // Ref https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten,
  // Default is 1
  int axis = 1;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "axis") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT)
          << "Node Attribute ERROR";
      axis = attributeProto.i();
    }
  }
  param->axis = axis;
  dstOp->main.value = param;
}

REGISTER_ONNX_NODE_PARSER(FlattenOnnx, Flatten);

}  // namespace parser
}  // namespace ace

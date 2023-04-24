#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(TransposeOnnx);

ace::OpType TransposeOnnx::opType() { return ace::OpType_Permute; }

ace::OpParameter TransposeOnnx::type() { return ace::OpParameter_Permute; }

void TransposeOnnx::parse(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                          std::vector<const onnx::TensorProto *> initializers) {
  auto param = new ace::PermuteT;

  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "perm") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS)
          << "Node Attribute ERROR";
      param->dims.resize(attributeProto.ints_size());
      for (int v = 0; v < attributeProto.ints_size(); ++v) {
        param->dims[v] = attributeProto.ints(v);
      }
    }
  }
  dstOp->main.value = param;
}

REGISTER_ONNX_NODE_PARSER(TransposeOnnx, Transpose);
}  // namespace parser
}  // namespace ace

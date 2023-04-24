#include <stdio.h>

#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(ReluOnnx);

ace::OpType ReluOnnx::opType() { return ace::OpType_ReLU; }
ace::OpParameter ReluOnnx::type() { return ace::OpParameter_Relu; }

void ReluOnnx::parse(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                     std::vector<const onnx::TensorProto*> initializers) {
  auto relu = new ace::ReluT;

  float slope = 0.01f;
  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();

    if (attributeName == "alpha") {
      DCHECK(attributeProto.type() ==
             ::onnx::AttributeProto_AttributeType_FLOAT)
          << "Node Attribute ERROR";
      slope = attributeProto.f();
    } else {
      DLOG(ERROR) << "TODO!";
    }
  }

  if (onnxNode->op_type() == "LeakyRelu") {
    relu->slope = slope;
  } else {
    relu->slope = .0f;
  }

  dstOp->main.value = relu;
}

REGISTER_ONNX_NODE_PARSER(ReluOnnx, Relu);
REGISTER_ONNX_NODE_PARSER(ReluOnnx, LeakyRelu);

}  // namespace parser
}  // namespace ace

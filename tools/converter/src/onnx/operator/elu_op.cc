#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(EluOnnx);

ace::OpType EluOnnx::opType() { return ace::OpType_ELU; }

ace::OpParameter EluOnnx::type() { return ace::OpParameter_ELU; }

void EluOnnx::parse(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                    std::vector<const onnx::TensorProto *> initializers) {
  auto eluParam = new ace::ELUT;

  float alpha = 1.0f;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "alpha") {
      alpha = attributeProto.f();
    }
  }

  eluParam->alpha = alpha;

  dstOp->main.value = eluParam;
}

REGISTER_ONNX_NODE_PARSER(EluOnnx, Elu);
}  // namespace parser
}  // namespace ace

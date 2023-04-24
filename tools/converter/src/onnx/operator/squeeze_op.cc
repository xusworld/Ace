#include <stdio.h>

#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(SqueezeOnnx);

ace::OpType SqueezeOnnx::opType() { return ace::OpType_Squeeze; }
ace::OpParameter SqueezeOnnx::type() { return ace::OpParameter_SqueezeParam; }

void SqueezeOnnx::parse(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                        std::vector<const onnx::TensorProto*> initializers) {
  auto para = new ace::SqueezeParamT;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "axes") {
      para->squeezeDims.resize(attributeProto.ints_size());
      for (int i = 0; i < para->squeezeDims.size(); ++i) {
        para->squeezeDims[i] = attributeProto.ints(i);
      }
    }
  }

  dstOp->main.value = para;
}

REGISTER_ONNX_NODE_PARSER(SqueezeOnnx, Squeeze);
}  // namespace parser
}  // namespace ace

#include <stdio.h>

#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(SpaceToDepthOnnx);

ace::OpType SpaceToDepthOnnx::opType() { return ace::OpType_SpaceToDepth; }

ace::OpParameter SpaceToDepthOnnx::type() {
  return ace::OpParameter_DepthSpaceParam;
}

void SpaceToDepthOnnx::parse(
    ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
    std::vector<const onnx::TensorProto*> initializers) {
  auto spaceToDepthParam = new ace::DepthSpaceParamT;

  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "blocksize") {
      spaceToDepthParam->blockSize = (int)attributeProto.i();
    }
  }

  dstOp->main.value = spaceToDepthParam;
}

REGISTER_ONNX_NODE_PARSER(SpaceToDepthOnnx, SpaceToDepth);
}  // namespace parser
}  // namespace ace

#include <ace/MNNDefine.h>
#include <stdio.h>

#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(DepthToSpaceOnnx);

ace::OpType DepthToSpaceOnnx::opType() { return ace::OpType_DepthToSpace; }

ace::OpParameter DepthToSpaceOnnx::type() {
  return ace::OpParameter_DepthSpaceParam;
}

void DepthToSpaceOnnx::parse(
    ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
    std::vector<const onnx::TensorProto*> initializers) {
  auto spaceToDepthParam = new ace::DepthSpaceParamT;

  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "blocksize") {
      spaceToDepthParam->blockSize = (int)attributeProto.i();
    } else if (attributeName == "mode") {
      std::map<const std::string, ace::DepthToSpaceMode> strToMode = {
          {"DCR", ace::DepthToSpaceMode_DCR},
          {"CRD", ace::DepthToSpaceMode_CRD}};
      const std::string& modeStr = attributeProto.s();
      if (strToMode.find(modeStr) != strToMode.end()) {
        spaceToDepthParam->mode = strToMode[modeStr];
      } else {
        MNN_ERROR("ONNX DepthToSpace mode [%s] is currently not supported.\n",
                  modeStr.c_str());
      }
    }
  }

  dstOp->main.value = spaceToDepthParam;
}

REGISTER_ONNX_NODE_PARSER(DepthToSpaceOnnx, DepthToSpace);

}  // namespace parser
}  // namespace ace

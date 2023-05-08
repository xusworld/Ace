#include <stdio.h>

#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(SplitOnnx);

ace::OpType SplitOnnx::opType() { return ace::OpType_Slice; }

ace::OpParameter SplitOnnx::type() { return ace::OpParameter_Slice; }

void SplitOnnx::parse(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      std::vector<const onnx::TensorProto*> initializers) {
  auto param = new ace::SliceT;
  int axis = 1;
  std::vector<int> slicePoints;
  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "axis") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT)
          << "Node Attribute ERROR";
      axis = attributeProto.i();
    } else if (attributeName == "split") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS)
          << "Node Attribute ERROR";
      const int splitSize = attributeProto.ints_size();
      for (int k = 0; k < splitSize; ++k) {
        slicePoints.push_back(attributeProto.ints(k));
      }
    }
  }
  param->axis = axis;
  param->slicePoints = slicePoints;
  param->sourceType = ace::FrontendFramework_TENSORFLOW;
  dstOp->main.value = param;
}

REGISTER_ONNX_NODE_PARSER(SplitOnnx, Split);

}  // namespace parser
}  // namespace ace

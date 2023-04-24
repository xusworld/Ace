
#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(ArgMaxOnnx);

void ArgMaxOnnx::parse(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                       std::vector<const onnx::TensorProto *> initializers) {
  auto axisT = new ace::ArgMaxT;
  int axis = 0;
  int keepdims = 1;
  int selectLastIndex = 0;  // Boolean value. Default to False.

  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();

    if (attributeName == "axis") {
      axis = attributeProto.i();
    }
    if (attributeName == "keepdims") {
      keepdims = attributeProto.i();
    }
    if (attributeName == "select_last_index") {
      // Ignored for now. argmax implementation does not support this yet.
      selectLastIndex = attributeProto.i();
    }
  }
  if (keepdims == 1) {
    LOG(FATAL)
        << "ONNX ArgMax with keepdims == true is currently not supported.";
  }
  axisT->axis = axis;
  axisT->topK = 1;
  axisT->outMaxVal = 0;
  dstOp->main.value = axisT;
}

ace::OpType ArgMaxOnnx::opType() { return ace::OpType_ArgMax; }

ace::OpParameter ArgMaxOnnx::type() { return ace::OpParameter_ArgMax; }

REGISTER_ONNX_NODE_PARSER(ArgMaxOnnx, ArgMax);

}  // namespace parser
}  // namespace ace

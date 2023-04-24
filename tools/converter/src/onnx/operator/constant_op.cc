#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(ConstantOnnx);

ace::OpType ConstantOnnx::opType() { return ace::OpType_Const; }
ace::OpParameter ConstantOnnx::type() { return ace::OpParameter_Blob; }

void ConstantOnnx::parse(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                         std::vector<const onnx::TensorProto *> initializers) {
  const onnx::TensorProto *constantTp;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "value") {
      constantTp = &attributeProto.t();
    }
  }
  if (!constantTp) {
    DLOG(FATAL) << "Constant No TensorProto Data!!!==> " << dstOp->name;
  }
  auto constantParam = OnnxTensorToBlob(constantTp);
  dstOp->main.value = constantParam;
  DCHECK(onnxNode->input_size() == 0)
      << "Constant Should Not Have Input!!! ===> " << dstOp->name;
}

REGISTER_ONNX_NODE_PARSER(ConstantOnnx, Constant);
}  // namespace parser
}  // namespace ace

#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(CosineSimilarityOnnx);

ace::OpType CosineSimilarityOnnx::opType() {
  return ace::OpType_CosineSimilarity;
}

ace::OpParameter CosineSimilarityOnnx::type() { return ace::OpParameter_NONE; }

void CosineSimilarityOnnx::parse(
    ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
    std::vector<const onnx::TensorProto *> initializers) {
  std::string type;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    auto att = onnxNode->attribute(i);
    if ("operator" == att.name()) {
      type = att.s();
      break;
    }
  }
  DCHECK(type == "cosine_similarity") << " NOT SUPPPRT";
  return;
}

REGISTER_ONNX_NODE_PARSER(CosineSimilarityOnnx, ATen);

}  // namespace parser
}  // namespace ace

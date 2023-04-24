#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(LRNOnnx);

ace::OpType LRNOnnx::opType() { return ace::OpType_LRN; }

ace::OpParameter LRNOnnx::type() { return ace::OpParameter_LRN; }

void LRNOnnx::parse(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                    std::vector<const onnx::TensorProto*> initializers) {
  auto param = new ace::LRNT;

  int size = 0;
  float alpha = 0.0001;
  float beta = 0.75;
  float bias = 1.0;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "size") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT)
          << "Node Attribute ERROR";
      size = attributeProto.i();
    } else if (attributeName == "alpha") {
      DCHECK(attributeProto.type() ==
             ::onnx::AttributeProto_AttributeType_FLOAT)
          << "Node Attribute ERROR";
      alpha = attributeProto.f();
    } else if (attributeName == "beta") {
      DCHECK(attributeProto.type() ==
             ::onnx::AttributeProto_AttributeType_FLOAT)
          << "Node Attribute ERROR";
      beta = attributeProto.f();
    } else if (attributeName == "bias") {
      DCHECK(attributeProto.type() ==
             ::onnx::AttributeProto_AttributeType_FLOAT)
          << "Node Attribute ERROR";
      bias = attributeProto.f();
    }
  }
  DCHECK(bias == 1.0) << "LRN bias must be 1.0";

  param->alpha = alpha;
  param->beta = beta;
  param->localSize = size;
  param->regionType = 0;
  dstOp->main.value = param;
}

REGISTER_ONNX_NODE_PARSER(LRNOnnx, LRN);

}  // namespace parser
}  // namespace ace

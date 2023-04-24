#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(CastOnnx);

ace::OpType CastOnnx::opType() { return ace::OpType_Cast; }
ace::OpParameter CastOnnx::type() { return ace::OpParameter_CastParam; }

void CastOnnx::parse(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                     std::vector<const onnx::TensorProto *> initializers) {
  std::unique_ptr<ace::CastParamT> castParam(new ace::CastParamT);

  // not to use srcT parameter!
  castParam->srcT = ace::DataType_MAX;

  ::onnx::TensorProto_DataType castTo = ::onnx::TensorProto_DataType_UNDEFINED;
  const int attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "to") {
      castTo = static_cast<::onnx::TensorProto_DataType>(attributeProto.i());
    }
  }

  castParam->dstT = ToAceDataType(castTo);
  dstOp->main.value = castParam.release();
}

REGISTER_ONNX_NODE_PARSER(CastOnnx, Cast);

}  // namespace parser
}  // namespace ace

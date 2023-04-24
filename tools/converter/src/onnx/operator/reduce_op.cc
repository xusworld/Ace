#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(ReduceOnnx);

ace::OpType ReduceOnnx::opType() { return ace::OpType_Reduction; }
ace::OpParameter ReduceOnnx::type() { return ace::OpParameter_ReductionParam; }

void ReduceOnnx::parse(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                       std::vector<const onnx::TensorProto *> initializers) {
  auto param = new ace::ReductionParamT;

  std::vector<int> axes;
  bool keepdims = true;
  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();

    if (attributeName == "axes") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS)
          << "Node Attribute ERROR";
      const int size = attributeProto.ints_size();
      for (int k = 0; k < size; ++k) {
        axes.push_back(attributeProto.ints(k));
      }
    } else if (attributeName == "keepdims") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT)
          << "Node Attribute ERROR";
      keepdims = static_cast<bool>(attributeProto.i());
    }
  }

  auto type = onnxNode->op_type();
  if (type == "ReduceMean") {
    param->operation = ace::ReductionType_MEAN;
  } else if (type == "ReduceMax") {
    param->operation = ace::ReductionType_MAXIMUM;
  } else if (type == "ReduceMin") {
    param->operation = ace::ReductionType_MINIMUM;
  } else if (type == "ReduceProd") {
    param->operation = ace::ReductionType_PROD;
  } else if (type == "ReduceSum") {
    param->operation = ace::ReductionType_SUM;
  } else if (type == "ReduceSumSquare") {
    param->operation = ace::ReductionType_SUMSQ;
  } else {
    DLOG(ERROR) << "TODO ==> " << type;
  }

  param->dType = ace::DataType_DT_FLOAT;
  param->dim = axes;
  param->keepDims = keepdims;
  dstOp->main.value = param;
}

REGISTER_ONNX_NODE_PARSER(ReduceOnnx, ReduceMean);
REGISTER_ONNX_NODE_PARSER(ReduceOnnx, ReduceMax);
REGISTER_ONNX_NODE_PARSER(ReduceOnnx, ReduceMin);
REGISTER_ONNX_NODE_PARSER(ReduceOnnx, ReduceProd);
REGISTER_ONNX_NODE_PARSER(ReduceOnnx, ReduceSum);
REGISTER_ONNX_NODE_PARSER(ReduceOnnx, ReduceSumSquare);

}  // namespace parser
}  // namespace ace

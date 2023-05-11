#include <stdio.h>

#include "../op_converter.h"
#include "ace/ir/op_generated.h"
#include "ace/ir/op_option_generated.h"
#include "ir/types_generated.h"

namespace ace {
namespace model {

DECLARE_ONNX_NODE_PARSER(ReshapeOnnxNodeParser);

void ReshapeOnnxNodeParser::parse(
    ace::OpT* op, const onnx::NodeProto* node,
    std::vector<const onnx::TensorProto*> initializers) {
  auto option = new ace::ReshapeOptionT;
  option->dimType = ace::DataFormat_NCHW;
  op->option.value = option;
}

ace::OpType ReshapeOnnxNodeParser::type() { return ace::OpType_Reshape; }

static OnnxNodeParserRegister<ReshapeOnnxNodeParser> _reshape_op_parser(
    "Reshape");

}  // namespace model
}  // namespace ace

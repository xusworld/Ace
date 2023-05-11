#include <stdio.h>

#include "../op_converter.h"
#include "ace/ir/op_generated.h"
#include "ace/ir/op_option_generated.h"
#include "ir/types_generated.h"

namespace ace {
namespace model {

DECLARE_ONNX_NODE_PARSER(Conv2DOnnxNodeParser);

void Conv2DOnnxNodeParser::parse(
    ace::OpT* op, const onnx::NodeProto* node,
    std::vector<const onnx::TensorProto*> initializers) {
  auto option = new ace::Pool2DOptionT;
  //   op->dimType = ace::DataFormat_NCHW;
  op->option.value = option;
}

ace::OpType Conv2DOnnxNodeParser::type() { return ace::OpType_Pool2D; }

static OnnxNodeParserRegister<Conv2DOnnxNodeParser> _conv2d_op_parser("Conv2D");

}  // namespace model
}  // namespace ace

#include <stdio.h>

#include "../op_converter.h"
#include "ace/ir/op_generated.h"
#include "ace/ir/op_option_generated.h"
#include "ir/types_generated.h"

namespace ace {
namespace model {

DECLARE_ONNX_NODE_PARSER(PoolOnnxNodeParser);

void PoolOnnxNodeParser::parse(
    ace::OpT* op, const onnx::NodeProto* node,
    std::vector<const onnx::TensorProto*> initializers) {
  auto option = new ace::Pool2DOptionT;
  //   op->dimType = ace::DataFormat_NCHW;
  op->option.value = option;
}

ace::OpType PoolOnnxNodeParser::type() { return ace::OpType_Pool2D; }

static OnnxNodeParserRegister<PoolOnnxNodeParser> _reshape_op_parser("Pool");

}  // namespace model
}  // namespace ace

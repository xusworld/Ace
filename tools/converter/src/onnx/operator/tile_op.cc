#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(TileOnnx);

ace::OpType TileOnnx::opType() { return ace::OpType_Tile; }

ace::OpParameter TileOnnx::type() { return ace::OpParameter_NONE; }

void TileOnnx::parse(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                     std::vector<const onnx::TensorProto *> initializers) {
  return;
}

REGISTER_ONNX_NODE_PARSER(TileOnnx, Tile);

}  // namespace parser
}  // namespace ace

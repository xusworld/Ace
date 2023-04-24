#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(GridSampleOnnx);

ace::OpType GridSampleOnnx::opType() { return ace::OpType_GridSample; }

ace::OpParameter GridSampleOnnx::type() { return ace::OpParameter_GridSample; }

void GridSampleOnnx::parse(
    ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
    std::vector<const onnx::TensorProto *> initializers) {
  auto gridSampleParam = new ace::GridSampleT;

  gridSampleParam->mode = ace::SampleMode_BILINEAR;
  gridSampleParam->paddingMode = ace::BorderMode_ZEROS;
  gridSampleParam->alignCorners = false;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "mode") {
      switch (attributeProto.i()) {
        case 0:
          gridSampleParam->mode = ace::SampleMode_BILINEAR;
          break;
        case 1:
          gridSampleParam->mode = ace::SampleMode_NEAREST;
          break;
        default:
          LOG(FATAL) << "Unknown mode for " << onnxNode->name() << "!";
          break;
      }
    }
    if (attributeName == "padding_mode") {
      switch (attributeProto.i()) {
        case 0:
          gridSampleParam->paddingMode = ace::BorderMode_ZEROS;
          break;
        case 1:
          gridSampleParam->paddingMode = ace::BorderMode_CLAMP;
          break;
        case 2:
          gridSampleParam->paddingMode = ace::BorderMode_REFLECTION;
          break;
        default:
          LOG(FATAL) << "Unknown padding mode for " << onnxNode->name() << "!";
          break;
      }
    }
    if (attributeName == "align_corners") {
      gridSampleParam->alignCorners = attributeProto.i();
    }
  }

  dstOp->main.value = gridSampleParam;
}

// REGISTER_ONNX_NODE_PARSER(GridSampleOnnx, GridSample);

// When we export torch.nn.functional.grid_sample to onnx, it's called
// GridSampler rather than GridSample, thus, we have to add the "r"
#define REGISTER_ONNX_NODE_PARSER_r(name, opType) \
  static OnnxNodeParserRegister<name> _Convert_##opType(#opType "r")
REGISTER_ONNX_NODE_PARSER_r(GridSampleOnnx, GridSample);

}  // namespace parser
}  // namespace ace

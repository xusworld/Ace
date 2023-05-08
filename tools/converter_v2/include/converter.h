#pragma once

#include "ace/ir/graph_generated.h"
#include "status.h"

namespace ace {
namespace model {

Status OnnxToAceModel(const std::string& onnx_model_path,
                      std::unique_ptr<ace::GraphProtoT>& graph_proto);
}  // namespace model
}  // namespace ace
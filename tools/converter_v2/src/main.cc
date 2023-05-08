#include <glog/logging.h>

#include <iostream>

#include "ace/ir/graph_generated.h"
#include "converter.h"

int main(int argc, char *argv[]) {
  LOG(INFO) << "Ace Model Converter V2" << std::endl;
  const std::string onnx_model_path =
      "/data/lukedong/Ace/models/mobilenetv2-7.onnx";
  const std::string ace_model_path =
      "/data/lukedong/Ace/models/mobilenetv2-7.ace";

  std::unique_ptr<ace::GraphProtoT> graph =
      std::unique_ptr<ace::GraphProtoT>(new ace::GraphProtoT());
  ace::model::OnnxToAceModel(onnx_model_path, graph);
  return 0;
}

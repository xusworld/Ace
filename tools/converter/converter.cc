#include <glog/logging.h>

#include <memory>

#include "converter.h"

int main() {
  LOG(INFO) << "Ace model parser, only support onnx model to ace model."
            << std::endl;
  const std::string model = "/data/lukedong/Ace/onnx/inception-v1-3.onnx";

  std::unique_ptr<ace::NetT> netT = std::unique_ptr<ace::NetT>(new ace::NetT());
  ace::converter::Onnx2AceNet(model, "1", netT);

  return 0;
}
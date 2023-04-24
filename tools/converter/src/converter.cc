#include <iostream>

#include "cli.hpp"
#include "config.hpp"

int main(int argc, char *argv[]) {
  std::cout << "Ace model parser..." << std::endl;
  modelConfig config;
  config.modelFile = "/data/lukedong/Ace/models/mobilenetv2-7.onnx";
  config.model = modelConfig::ONNX;
  config.MNNModel = "/data/lukedong/Ace/models/mobilenetv2-7.ace";
  ace::Cli::convertModel(config);
  return 0;
}

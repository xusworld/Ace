//
//  cli.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cli.hpp"
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <unistd.h>
#endif
#include <ace/MNNDefine.h>
#include <glog/logging.h>

#include "PostConverter.hpp"
#include "ace_generated.h"
#include "config.hpp"
#include "cxxopts.hpp"
#include "onnxConverter.hpp"
#include "writeFb.hpp"

namespace ace {

static float gMNNVersion = 1.2f;

bool Cli::initializeMNNConvertArgs(modelConfig& config, int argc, char** argv) {
  /*
  std::cout << "init... " << std::endl;
  cxxopts::Options options("MNNConvert");

  options.positional_help("[optional args]").show_positional_help();

  options.allow_unrecognised_options().add_options()(
      std::make_pair("h", "help"), "Convert Other Model Format To MNN Model\n")(
      std::make_pair("v", "version"), "show current version")(
      std::make_pair("f", "framework"),
      "model type, ex: [TF,CAFFE,ONNX,TFLITE,MNN]",
      cxxopts::value<std::string>())(
      "modelFile", "tensorflow Pb or caffeModel, ex: *.pb,*caffemodel",
      cxxopts::value<std::string>())(
      "batch",
      "if model input's batch is not set, set as the batch size you set",
      cxxopts::value<int>())(
      "optimizeLevel",
      "graph optimize option, 1: use graph optimize only for every input case "
      "is right, 2: normally right but some case may be wrong, default 1",
      cxxopts::value<int>())(
      "optimizePrefer",
      "graph optimize option, 0 for normal, 1 for smalleset, 2 for fastest",
      cxxopts::value<int>())("prototxt", "only used for caffe, ex: *.prototxt",
                             cxxopts::value<std::string>())(
      "MNNModel", "MNN model, ex: *.mnn", cxxopts::value<std::string>())(
      "fp16", "save Conv's weight/bias in half_float data type")(
      "benchmarkModel",
      "Do NOT save big size data, such as Conv's weight,BN's gamma,beta,mean "
      "and variance etc. Only used to test "
      "the cost of the model")("debug", "Enable debugging mode.")(
      "forTraining",
      "whether or not to save training ops BN and Dropout, default: false",
      cxxopts::value<bool>())(
      "weightQuantBits",
      "save conv/matmul/LSTM float weights to int8 type, only optimize for "
      "model size, 2-8 bits, default: 0, which means no weight quant",
      cxxopts::value<int>())(
      "weightQuantAsymmetric",
      "the default weight-quant uses SYMMETRIC quant method, which is "
      "compatible with old MNN versions. "
      "you can try set --weightQuantAsymmetric to use asymmetric quant method "
      "to improve accuracy of the weight-quant model in some cases, "
      "but asymmetric quant model cannot run on old MNN versions. You will "
      "need to upgrade MNN to new version to solve this problem. default: "
      "false",
      cxxopts::value<bool>())(
      "compressionParamsFile",
      "The path of the compression parameters that stores activation, "
      "weight scales and zero points for quantization or information "
      "for sparsity.",
      cxxopts::value<std::string>())(
      "saveStaticModel", "save static model with fix shape, default: false",
      cxxopts::value<bool>())("targetVersion",
                              "compability for old mnn engine, default: 1.2f",
                              cxxopts::value<float>())(
      "inputConfigFile",
      "set input config file for static model, ex: ~/config.txt",
      cxxopts::value<std::string>());

  auto result = options.parse(argc, argv);

  std::cout << "help... " << std::endl;
  if (result.count("help")) {
    std::cout << options.help({""}) << std::endl;
    return false;
  }

  if (result.count("version")) {
    std::cout << gMNNVersion << std::endl;
    return false;
  }

  config.model = config.MAX_SOURCE;
  // model source
  if (result.count("framework")) {
    const std::string frameWork = result["framework"].as<std::string>();
    if ("ONNX" == frameWork) {
      config.model = modelConfig::ONNX;
    } else if ("MNN" == frameWork) {
      config.model = modelConfig::MNN;
    } else {
      std::cout << "Framework Input ERROR or Not Support This Model Type Now!"
                << std::endl;
      return false;
    }
  } else {
    std::cout << options.help({""}) << std::endl;
    DLOG(INFO) << "framework Invalid, use -f CAFFE/ace/ONNX/TFLITE/TORCH !";
    return false;
  }

  // model file path
  if (result.count("modelFile")) {
    const std::string modelFile = result["modelFile"].as<std::string>();
    if (CommonKit::FileIsExist(modelFile)) {
      config.modelFile = modelFile;
    } else {
      DLOG(INFO) << "Model File Does Not Exist! ==> " << modelFile;
      return false;
    }
  } else {
    DLOG(INFO) << "modelFile Not set Invalid, use --modelFile to set!";
    return false;
  }
  // Optimize Level
  if (result.count("optimizeLevel")) {
    config.optimizeLevel = result["optimizeLevel"].as<int>();
    if (config.optimizeLevel > 1) {
      DLOG(INFO) << "\n optimizeLevel > 1, some case may be wrong";
    }
  }

  // prototxt file path
  if (result.count("prototxt")) {
    const std::string prototxt = result["prototxt"].as<std::string>();
    if (CommonKit::FileIsExist(prototxt)) {
      config.prototxtFile = prototxt;
    } else {
      DLOG(INFO) << "Proto File Does Not Exist!";
      return false;
    }
  } else {
    // caffe model must have this option
    if (config.model == config.CAFFE) {
      DLOG(INFO)
          << "Proto File Not Set, use --prototxt XXX.prototxt to set it!";
      return false;
    }
  }

  // MNN model output path
  if (result.count("MNNModel")) {
    const std::string MNNconfig = result["MNNModel"].as<std::string>();
    config.MNNModel = MNNconfig;
  } else {
    DLOG(INFO)
        << "MNNModel File Not Set, use --MNNModel XXX.prototxt to set it!";
    return false;
  }
  if (result.count("targetVersion")) {
    auto version = result["targetVersion"].as<float>();
    std::cout << "TargetVersion is " << version << std::endl;
    config.targetVersion = version;
  }
  // add MNN bizCode
  if (result.count("bizCodbizCodee")) {
    const std::string bizCode = result["bizCode"].as<std::string>();
    config.bizCode = bizCode;
  } else {
    MNN_ERROR("Don't has bizCode, use MNNTest for default\n");
    config.bizCode = "MNNTest";
  }

  // input config file path
  if (result.count("inputConfigFile")) {
    const std::string inputConfigFile =
        result["inputConfigFile"].as<std::string>();
    config.inputConfigFile = inputConfigFile;
  }

  // benchmarkModel
  if (result.count("benchmarkModel")) {
    config.benchmarkModel = true;
    config.bizCode = "benchmark";
  }
  // half float
  if (result.count("fp16")) {
    config.saveHalfFloat = true;
  }
  if (result.count("forTraining")) {
    config.forTraining = true;
  }
  if (result.count("batch")) {
    config.defaultBatchSize = result["batch"].as<int>();
  }
  if (result.count("weightQuantBits")) {
    config.weightQuantBits = result["weightQuantBits"].as<int>();
  }
  if (result.count("weightQuantAsymmetric")) {
    config.weightQuantAsymmetric = true;
  }
  if (result.count("saveStaticModel")) {
    config.saveStaticModel = true;
  }
  if (result.count("optimizePrefer")) {
    config.optimizePrefer = result["optimizePrefer"].as<int>();
  }
  // Int8 calibration table path.
  if (result.count("compressionParamsFile")) {
    config.compressionParamsFile =
        result["compressionParamsFile"].as<std::string>();
  }
*/
  return true;
}

bool Cli::convertModel(modelConfig& config) {
  std::cout << "Start to Convert Other Model Format To Ace Model..."
            << std::endl;
  std::unique_ptr<ace::NetT> netT = std::unique_ptr<ace::NetT>(new ace::NetT());

  if (config.model == modelConfig::ONNX) {
    ace::parser::OnnxToAceModel(config.modelFile, config.bizCode, netT);
  } else {
    std::cout << "Not Support Model Type" << std::endl;
  }
  int error = 0;
  if (config.defaultBatchSize > 0) {
    for (const auto& op : netT->oplists) {
      if (op->type != OpType_Input || nullptr == op->main.AsInput()) {
        continue;
      }
      auto inputP = op->main.AsInput();
      if (inputP->dims.size() >= 1 && inputP->dims[0] <= 0) {
        std::cout << "Set " << op->name
                  << " batch = " << config.defaultBatchSize << std::endl;
        inputP->dims[0] = config.defaultBatchSize;
      }
    }
  }
  if (config.model != modelConfig::MNN) {
    LOG(INFO) << "Start to Optimize Ace Model...";
    std::unique_ptr<ace::NetT> newNet =
        optimizeNet(netT, config.forTraining, config);
    error = writeFb(newNet, config.MNNModel, config);
  } else {
    error = writeFb(netT, config.MNNModel, config);
  }
  if (0 == error) {
    std::cout << "Converted Success!" << std::endl;
  } else {
    std::cout << "Converted Failed!" << std::endl;
  }
  return true;
}
};  // namespace ace

bool CommonKit::FileIsExist(std::string path) {
#if defined(_MSC_VER)
  if (INVALID_FILE_ATTRIBUTES != GetFileAttributes(path.c_str()) &&
      GetLastError() != ERROR_FILE_NOT_FOUND) {
    return true;
  }
#else
  if ((access(path.c_str(), F_OK)) != -1) {
    return true;
  }
#endif
  return false;
}

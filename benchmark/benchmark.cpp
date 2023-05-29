#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#endif

#include <MNN/MNNDefine.h>

#include "core/Interpreter.hpp"
#include "core/device.h"
#include "core/tensor.h"
#include "glog/logging.h"
#include "revertMNNModel.hpp"

/**
 TODOs:
 1. dynamically get CPU related info.
 2. iOS support
 */
struct Model {
  std::string name;
  std::string model_file;
};

#if !defined(_MSC_VER)
inline bool file_exist(const char* file) {
  struct stat buffer;
  return stat(file, &buffer) == 0;
}
#endif

std::vector<Model> findModelFiles(const char* dir) {
  std::vector<Model> models;

  DIR* root;
  if ((root = opendir(dir)) == NULL) {
    std::cout << "open " << dir << " failed: " << strerror(errno) << std::endl;
    return models;
  }

  struct dirent* ent;
  while ((ent = readdir(root)) != NULL) {
    Model m;
    if (ent->d_name[0] != '.') {
      m.name = ent->d_name;
      m.model_file = std::string(dir) + "/" + m.name;
      if (file_exist(m.model_file.c_str())) {
        models.push_back(std::move(m));
      }
    }
    break;
  }
  closedir(root);

  return models;
}

void setInputData(tars::Tensor* tensor) {
  float* data = tensor->host<float>();
  Revert::fillRandValue(data, tensor->elementSize());
}

static inline uint64_t getTimeInUs() {
  uint64_t time;
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
  return time;
}

std::vector<float> doBench(Model& model, int loop, int warmup = 10,
                           int forward = MNN_FORWARD_CPU,
                           bool only_inference = true, int numberThread = 4,
                           int precision = 2, float sparsity = 0.0f,
                           int sparseBlockOC = 1) {
  auto revertor = std::unique_ptr<Revert>(new Revert(model.model_file.c_str()));
  revertor->initialize(sparsity, sparseBlockOC);
  auto modelBuffer = revertor->getBuffer();
  const auto bufferSize = revertor->getBufferSize();
  // 创建 Interpreter
  auto net = std::shared_ptr<tars::Interpreter>(
      tars::Interpreter::createFromBuffer(modelBuffer, bufferSize));

  revertor.reset();
  // 设置 Session 模式
  net->setSessionMode(tars::Interpreter::Session_Release);
  // ScheduleConfig 是一个全局纬度的配置信息
  tars::ScheduleConfig config;
  config.numThread = numberThread;
  config.type = static_cast<MNNForwardType>(forward);
  // Device Config
  tars::BackendConfig backendConfig;
  // 下面的配置都可以删除
  backendConfig.precision = (tars::BackendConfig::PrecisionMode)precision;
  backendConfig.power = tars::BackendConfig::Power_High;
  config.backendConfig = &backendConfig;

  std::vector<float> costs;
  LOG(INFO) << "create the session";
  tars::Session* session = net->createSession(config);
  tars::Tensor* input = net->getSessionInput(session, NULL);

  // if the model has not the input dimension, umcomment the below code to set
  // the input dims std::vector<int> dims{1, 3, 224, 224};
  // net->resizeTensor(input, dims);
  // net->resizeSession(session);

  LOG(INFO) << "release model";
  net->releaseModel();

  const tars::Device* inBackend = net->getBackend(session, input);

  LOG(INFO) << "create host tensor";
  std::shared_ptr<tars::Tensor> givenTensor(
      tars::Tensor::createHostTensorFromDevice(input, false));

  LOG(INFO) << "session output";
  auto outputTensor = net->getSessionOutput(session, NULL);

  LOG(INFO) << "create host tensor";
  std::shared_ptr<tars::Tensor> expectTensor(
      tars::Tensor::createHostTensorFromDevice(outputTensor, false));

  LOG(INFO) << "warm up ...";
  // Warming up...
  for (int i = 0; i < warmup; ++i) {
    LOG(INFO) << "warm up " << i << "th iter";
    void* host =
        input->map(tars::Tensor::MAP_TENSOR_WRITE, input->getDimensionType());
    input->unmap(tars::Tensor::MAP_TENSOR_WRITE, input->getDimensionType(),
                 host);

    net->runSession(session);

    host = outputTensor->map(tars::Tensor::MAP_TENSOR_READ,
                             outputTensor->getDimensionType());
    outputTensor->unmap(tars::Tensor::MAP_TENSOR_READ,
                        outputTensor->getDimensionType(), host);
  }

  LOG(INFO) << "infer ...";
  for (int round = 0; round < loop; round++) {
    LOG(INFO) << "infer " << round << "th iter";
    auto timeBegin = getTimeInUs();
    void* host =
        input->map(tars::Tensor::MAP_TENSOR_WRITE, input->getDimensionType());
    input->unmap(tars::Tensor::MAP_TENSOR_WRITE, input->getDimensionType(),
                 host);
    net->runSession(session);
    host = outputTensor->map(tars::Tensor::MAP_TENSOR_READ,
                             outputTensor->getDimensionType());
    outputTensor->unmap(tars::Tensor::MAP_TENSOR_READ,
                        outputTensor->getDimensionType(), host);
    auto timeEnd = getTimeInUs();
    costs.push_back((timeEnd - timeBegin) / 1000.0);
  }
  return costs;
}

void displayStats(const std::string& name, const std::vector<float>& costs) {
  float max = 0, min = FLT_MAX, sum = 0, avg;
  for (auto v : costs) {
    max = fmax(max, v);
    min = fmin(min, v);
    sum += v;
    // printf("[ - ] cost：%f ms\n", v);
  }
  avg = costs.size() > 0 ? sum / costs.size() : 0;
  printf("[ - ] %-24s    max = %8.3f ms  min = %8.3f ms  avg = %8.3f ms\n",
         name.c_str(), max, avg == 0 ? 0 : min, avg);
}
static inline std::string forwardType(MNNForwardType type) {
  switch (type) {
    case MNN_FORWARD_CPU:
      return "CPU";
    default:
      break;
  }
  return "N/A";
}

int main(int argc, const char* argv[]) {
  std::cout << "MNN benchmark" << std::endl;
  int loop = 10;
  int warmup = 10;
  MNNForwardType forward = MNN_FORWARD_CPU;
  int numberThread = 4;
  int precision = 2;
  float sparsity = 0.0f;
  int sparseBlockOC = 1;
  if (argc <= 2) {
    std::cout << "Usage: " << argv[0]
              << " models_folder [loop_count] [warmup] [forwardtype] "
                 "[numberThread] [precision] [weightSparsity]"
              << std::endl;
    return 1;
  }
  if (argc >= 3) {
    loop = atoi(argv[2]);
  }
  if (argc >= 4) {
    warmup = atoi(argv[3]);
  }
  if (argc >= 5) {
    forward = static_cast<MNNForwardType>(atoi(argv[4]));
  }
  if (argc >= 6) {
    numberThread = atoi(argv[5]);
  }

  if (argc >= 7) {
    precision = atoi(argv[6]);
  }

  if (argc >= 8) {
    sparsity = atof(argv[7]);
  }

  if (argc >= 9) {
    sparseBlockOC = atoi(argv[8]);
  }

  std::cout << "Forward type: **" << forwardType(forward)
            << "** thread=" << numberThread << "** precision=" << precision
            << "** sparsity=" << sparsity
            << "** sparseBlockOC=" << sparseBlockOC << std::endl;
  std::vector<Model> models = findModelFiles(argv[1]);

  std::cout << "--------> Benchmarking... loop = " << argv[2]
            << ", warmup = " << warmup << std::endl;

  for (auto& m : models) {
    std::vector<float> costs =
        doBench(m, loop, warmup, forward, false, numberThread, precision,
                sparsity, sparseBlockOC);
    displayStats(m.name, costs);
  }
}

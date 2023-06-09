//
//  testModel.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define MNN_OPEN_TIME_TRACE

#include <MNN/MNNDefine.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <MNN/AutoTime.hpp>
#include <fstream>
#include <map>
#include <sstream>

#include "core/Interpreter.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/device.h"
#include "core/tensor.h"

#define NONE "\e[0m"
#define RED "\e[0;31m"
#define GREEN "\e[0;32m"
#define L_GREEN "\e[1;32m"
#define BLUE "\e[0;34m"
#define L_BLUE "\e[1;34m"
#define BOLD "\e[1m"

template <typename T>
inline T stringConvert(const char* number) {
  std::istringstream os(number);
  T v;
  os >> v;
  return v;
}

tars::Tensor* createTensor(const tars::Tensor* shape, const char* path) {
  std::ifstream stream(path);
  if (stream.fail()) {
    return NULL;
  }

  auto result = new tars::Tensor(shape, shape->getDimensionType());
  auto data = result->host<float>();
  for (int i = 0; i < result->elementSize(); ++i) {
    double temp = 0.0f;
    stream >> temp;
    data[i] = temp;
  }
  stream.close();
  return result;
}

int main(int argc, const char* argv[]) {
  // check given & expect
  const char* modelPath = argv[1];
  const char* givenName = argv[2];
  const char* expectName = argv[3];
  MNN_PRINT("Testing model %s, input: %s, output: %s\n", modelPath, givenName,
            expectName);

  // create net
  auto type = MNN_FORWARD_CPU;
  if (argc > 4) {
    type = (MNNForwardType)stringConvert<int>(argv[4]);
  }
  auto tolerance = 0.1f;
  if (argc > 5) {
    tolerance = stringConvert<float>(argv[5]);
  }
  tars::BackendConfig::PrecisionMode precision =
      tars::BackendConfig::Precision_High;
  if (argc > 6) {
    precision = (tars::BackendConfig::PrecisionMode)stringConvert<int>(argv[6]);
  }
  std::shared_ptr<tars::Interpreter> net = std::shared_ptr<tars::Interpreter>(
      tars::Interpreter::createFromFile(modelPath),
      [](void* net) { tars::Interpreter::destroy((tars::Interpreter*)net); });

  // create session
  tars::ScheduleConfig config;
  config.type = type;
  tars::BackendConfig backendConfig;
  backendConfig.precision = precision;
  config.backendConfig = &backendConfig;
  auto session = net->createSession(config);

  // input dims
  std::vector<int> inputDims;
  if (argc > 7) {
    std::string inputShape(argv[7]);
    const char* delim = "x";
    std::ptrdiff_t p1 = 0, p2;
    while (1) {
      p2 = inputShape.find(delim, p1);
      if (p2 != std::string::npos) {
        inputDims.push_back(atoi(inputShape.substr(p1, p2 - p1).c_str()));
        p1 = p2 + 1;
      } else {
        inputDims.push_back(atoi(inputShape.substr(p1).c_str()));
        break;
      }
    }
  }
  for (auto dim : inputDims) {
    MNN_PRINT("%d ", dim);
  }
  MNN_PRINT("\n");

  auto allInput = net->getSessionInputAll(session);
  for (auto& iter : allInput) {
    auto inputTensor = iter.second;

    if (!inputDims.empty()) {
      MNN_PRINT("===========> Resize Tensor...\n");
      net->resizeTensor(inputTensor, inputDims);
      net->resizeSession(session);
    }

    auto size = inputTensor->size();
    if (size <= 0) {
      continue;
    }

    void* host = inputTensor->map(tars::Tensor::MAP_TENSOR_WRITE,
                                  inputTensor->getDimensionType());
    if (host != nullptr) {
      ::memset(host, 0, inputTensor->size());
    }
    inputTensor->unmap(tars::Tensor::MAP_TENSOR_WRITE,
                       inputTensor->getDimensionType(), host);
  }

  // write input tensor
  auto inputTensor = net->getSessionInput(session, NULL);
  std::shared_ptr<tars::Tensor> givenTensor(
      createTensor(inputTensor, givenName),
      [](void* t) { tars::Tensor::destroy((tars::Tensor*)t); });
  if (!givenTensor) {
#if defined(_MSC_VER)
    printf("Failed to open input file %s.\n", givenName);
#else
    printf(RED "Failed to open input file %s.\n" NONE, givenName);
#endif
    return -1;
  }
  // First time
  void* host = inputTensor->map(tars::Tensor::MAP_TENSOR_WRITE,
                                givenTensor.get()->getDimensionType());
  if (host != nullptr) {
    ::memcpy(host, givenTensor->host<uint8_t>(), givenTensor->size());
  }
  inputTensor->unmap(tars::Tensor::MAP_TENSOR_WRITE,
                     givenTensor.get()->getDimensionType(), host);

  // infer
  net->runSession(session);
  // read expect tensor
  auto outputTensor = net->getSessionOutput(session, NULL);
  std::shared_ptr<tars::Tensor> expectTensor(
      createTensor(outputTensor, expectName));
  if (!expectTensor.get()) {
#if defined(_MSC_VER)
    printf("Failed to open expect file %s.\n", expectName);
#else
    printf(RED "Failed to open expect file %s.\n" NONE, expectName);
#endif
    return -1;
  }

  // compare output with expect
  bool correct = tars::TensorUtils::compareTensors(
      outputTensor, expectTensor.get(), tolerance, true);
  if (!correct) {
#if defined(_MSC_VER)
    printf("Test Failed %s!\n", modelPath);
#else
    printf(RED "Test Failed %s!\n" NONE, modelPath);
#endif
    return -1;
  } else {
    printf("First run pass\n");
  }
  // Run Second time
  void* host1 = inputTensor->map(tars::Tensor::MAP_TENSOR_WRITE,
                                 givenTensor.get()->getDimensionType());
  if (host1 != nullptr) {
    ::memcpy(host1, givenTensor->host<uint8_t>(), givenTensor->size());
  }
  inputTensor->unmap(tars::Tensor::MAP_TENSOR_WRITE,
                     givenTensor.get()->getDimensionType(), host1);

  // infer
  net->runSession(session);
  // read expect tensor
  std::shared_ptr<tars::Tensor> expectTensor2(
      createTensor(outputTensor, expectName));
  correct = tars::TensorUtils::compareTensors(outputTensor, expectTensor2.get(),
                                              tolerance, true);

  if (correct) {
#if defined(_MSC_VER)
    printf("Test %s Correct!\n", modelPath);
#else
    printf(GREEN BOLD "Test %s Correct!\n" NONE, modelPath);
#endif
  } else {
#if defined(_MSC_VER)
    printf("Test Failed %s!\n", modelPath);
#else
    printf(RED "Test Failed %s!\n" NONE, modelPath);
#endif
  }
  return 0;
}

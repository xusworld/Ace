//
//  transformerDemo.cpp
//  MNN
//
//  Created by MNN on b'2020/10/05'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Module.hpp>
#define MNN_OPEN_TIME_TRACE
#include <stdio.h>
#include <string.h>

#include <MNN/AutoTime.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <fstream>
#include <sstream>
using namespace tars::Express;
using namespace tars;
using namespace std;

int main(int argc, const char* argv[]) {
  int testSpeedTime = 10;
  if (argc < 2) {
    MNN_ERROR("Don't has model name\n");
    return 0;
  }
  BackendConfig config;
  //    Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU,
  //    config, 4);
  auto modelName = argv[1];
  std::shared_ptr<Module> model;
  Module::Config mdconfig;
  mdconfig.rearrange = true;
  model.reset(Module::load({"NmtModel/Placeholder", "NmtModel/Placeholder_1"},
                           {"NmtModel/transpose_2"}, modelName, &mdconfig));
  std::vector<int> input0 = {4405, 17, 235, 2441, 8, 27, 1969, 0};
  std::vector<int> input1 = {1, 1, 1, 1, 1, 1, 1, 1};
  auto first = _Input({8, 1}, NHWC, halide_type_of<int>());
  ::memcpy(first->writeMap<int>(), input0.data(), input0.size() * sizeof(int));
  auto second = _Input({8, 1}, NHWC, halide_type_of<int>());
  ::memcpy(second->writeMap<int>(), input1.data(), input1.size() * sizeof(int));
  std::vector<VARP> outputs;
  for (int i = 0; i < 2; ++i) {
    {
      AUTOTIME;
      Executor::getGlobalExecutor()->resetProfile();
      outputs = model->onForward({first, second});
      Executor::getGlobalExecutor()->dumpProfile();
    }
    std::ostringstream fileNameOs;
    std::ostringstream dimInfo;
    fileNameOs << i << "_output.txt";
    auto info = outputs[0]->getInfo();
    for (int d = 0; d < info->dim.size(); ++d) {
      dimInfo << info->dim[d] << "_";
    }
    auto fileName = fileNameOs.str();
    MNN_PRINT("Output Name: %s, Dim: %s\n", fileName.c_str(),
              dimInfo.str().c_str());
    auto ptr = outputs[0]->readMap<int>();
    std::ofstream outputOs(fileName.c_str());
    for (int i = 0; i < info->size; ++i) {
      outputOs << ptr[i] << "\n";
    }
  }
  for (int i = 0; i < testSpeedTime; ++i) {
    AUTOTIME;
    outputs = model->onForward({first, second});
  }

  return 0;
}

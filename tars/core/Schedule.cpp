#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>
#include <vector>

#include "core/Macro.h"
#include "core/RuntimeFactory.hpp"
#include "core/Schedule.hpp"
#include "core/TensorUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "glog/logging.h"
#include "shape/SizeComputer.hpp"
#include "utils/InitNet.hpp"

// #define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
using namespace std;
// #define MNN_AUTO_CHECK_COST
namespace tars {

MNNForwardType Schedule::getApprociateType(const ScheduleConfig& config) {
  MNNForwardType type = config.type;
  // FIXME: Support Auto determine
  if (MNN_FORWARD_AUTO == config.type) {
    // Define Auto choose priority
    std::vector<MNNForwardType> priorityList;
    priorityList.push_back(MNN_FORWARD_CUDA);  // CUDA
    priorityList.push_back(MNN_FORWARD_CPU);   // CPU

    for (auto bn : priorityList) {
      if (MNNGetExtraRuntimeCreator(bn) != nullptr) {
        type = (MNNForwardType)bn;
        break;
      }
    }
  }
  auto creator = MNNGetExtraRuntimeCreator(type);
  if (nullptr == creator) {
    MNN_PRINT("Can't Find type=%d backend, use %d instead\n", type,
              config.backupType);
    type = config.backupType;
  }

  return type;
}

static void generateScheduleGraph(
    vector<const Op*>& ops, const Net* net, const ScheduleConfig& configs,
    const vector<shared_ptr<Tensor>>& allTensors) {
  // for (int i = 0; i < net->oplists()->size(); ++i) {
  //     auto op       = net->oplists()->Get(i);
  //     MNN_PRINT("generateScheduleGraph, op type:%s, op name:%s\n",
  //     EnumNameOpType(op->type()), op->name()->c_str());
  // }

  if (configs.path.inputs.empty() && configs.path.outputs.empty()) {
    LOG(INFO) << "Use Default Linear schedule";
    // Use Default Linear schedule
    ops.clear();
    ops.reserve(net->oplists()->size());
    for (int i = 0; i < net->oplists()->size(); ++i) {
      auto op = net->oplists()->GetAs<Op>(i);
      ops.emplace_back(op);
    }
    return;
  }
  // 0: not set, 1: output, 2:input
  std::vector<int> tensorMask(net->tensorName()->size());
  ::memset(tensorMask.data(), 0, tensorMask.size() * sizeof(int));

  // 0: use, 1: no use
  std::vector<int> opMask(net->oplists()->size());
  ::memset(opMask.data(), 0, opMask.size() * sizeof(int));

  // Set Initial Status
  std::set<std::string> inputNames;
  std::set<std::string> outputNames;
  for (auto& n : configs.path.inputs) {
    inputNames.insert(n);
  }
  for (auto& n : configs.path.outputs) {
    outputNames.insert(n);
  }
  if (configs.path.mode == ScheduleConfig::Path::Mode::Tensor) {
    for (int i = 0; i < tensorMask.size(); ++i) {
      auto name = net->tensorName()->GetAsString(i)->c_str();
      if (outputNames.find(name) != outputNames.end()) {
        tensorMask[i] = 1;
      }
      // If both input/output, set as input
      if (inputNames.find(name) != inputNames.end()) {
        tensorMask[i] = 2;
      }
    }
  } else {
    // Op Mode
    for (int i = 0; i < opMask.size(); ++i) {
      auto op = net->oplists()->GetAs<Op>(i);
      if (nullptr == op->name()) {
        continue;
      }
      auto name = op->name()->c_str();
      if (outputNames.find(name) != outputNames.end()) {
        opMask[i] = 1;
        if (nullptr != op->outputIndexes()) {
          for (int j = 0; j < op->outputIndexes()->size(); ++j) {
            auto index = op->outputIndexes()->data()[j];
            if (tensorMask[index] != 2) {
              tensorMask[index] = 1;
            }
          }
        }
        if (nullptr != op->inputIndexes()) {
          for (int j = 0; j < op->inputIndexes()->size(); ++j) {
            auto index = op->inputIndexes()->data()[j];
            if (tensorMask[index] != 2) {
              tensorMask[index] = 1;
            }
          }
        }
      }
      if (inputNames.find(name) != inputNames.end()) {
        opMask[i] = 1;
        if (nullptr != op->outputIndexes()) {
          for (int j = 0; j < op->outputIndexes()->size(); ++j) {
            auto index = op->outputIndexes()->data()[j];
            tensorMask[index] = 2;
          }
        }
      }
    }
  }

  bool change = false;
  do {
    change = false;
    for (int i = 0; i < opMask.size(); ++i) {
      if (opMask[i] > 0) {
        continue;
      }
      auto op = net->oplists()->GetAs<Op>(i);
      if (nullptr != op->outputIndexes()) {
        for (int j = 0; j < op->outputIndexes()->size(); ++j) {
          auto index = op->outputIndexes()->data()[j];
          if (tensorMask[index] == 1) {
            opMask[i] = 1;
            change = true;
          }
        }
      }
      if (nullptr != op->inputIndexes() && opMask[i]) {
        for (int j = 0; j < op->inputIndexes()->size(); ++j) {
          auto index = op->inputIndexes()->data()[j];
          if (tensorMask[index] != 2) {
            tensorMask[index] = 1;
          }
        }
      }
    }
  } while (change);

  for (int i = 0; i < opMask.size(); ++i) {
    if (opMask[i] > 0) {
      ops.emplace_back(net->oplists()->GetAs<Op>(i));
    }
  }
}

static vector<Schedule::OpCacheInfo> _scheduleUnit(
    const Net* net, const ScheduleConfig& configs,
    const vector<shared_ptr<Tensor>>& allTensors) {
  vector<Schedule::OpCacheInfo> oplists;
  vector<const Op*> ops;
  generateScheduleGraph(ops, net, configs, allTensors);
  initPipelineInfosFromOps(oplists, ops, allTensors);
  return oplists;
}

bool Schedule::schedule(ScheduleInfo& scheduleInfo, const Net* net,
                        const std::vector<ScheduleConfig>& configs,
                        const RuntimeInfo& runtimeInfo) {
  if (nullptr == net->oplists()) {
    LOG(INFO) << "Empty net for schedule";
    return false;
  }

  // TODO 考虑删除 schedule 相关的设计
  if (scheduleInfo.defaultBackend.get() == nullptr &&
      scheduleInfo.allTensors.empty()) {
    // Const not init, init it
    BackendConfig defaultConfig;
    defaultConfig.flags = 4;
    scheduleInfo.defaultBackend.reset(
        runtimeInfo.second->onCreate(&defaultConfig));
    Status code = Status::OK();
    // 初始化 const tensors
    initConstTensors(scheduleInfo.allTensors, net,
                     scheduleInfo.defaultBackend.get(), code);
  }
  // 初始化 tensors
  bool valid = initTensors(scheduleInfo.allTensors, net);

  scheduleInfo.validForResize = valid;
  std::vector<std::shared_ptr<Tensor>>& allTensors = scheduleInfo.allTensors;

  std::vector<
      std::pair<Schedule::BackendCache, std::vector<Schedule::OpCacheInfo>>>
      result;

  for (auto& config : configs) {
    Device::Info compute;
    compute.type = getApprociateType(config);
    compute.numThread = config.numThread;
    compute.user = config.backendConfig;
    auto oplists = _scheduleUnit(net, config, allTensors);
    Schedule::BackendCache cache;
    cache.info = std::move(compute);
    result.emplace_back(std::make_pair(cache, std::move(oplists)));
  }

  scheduleInfo.pipelineInfo = std::move(result);

  // get all used op's output, drop unused op, won't change op order. always
  // insert all Input Ops
  std::vector<const Op*> oplists;
  {
    for (std::pair<Schedule::BackendCache, vector<Schedule::OpCacheInfo>>&
             pipeline : scheduleInfo.pipelineInfo) {
      for (auto& info : pipeline.second) {
        oplists.push_back(info.op);
      }
    }
  }
  // set tensors' input/output usage by oplists info
  setInputOutputForOps(allTensors, oplists,
                       net->usage() == Usage_INFERENCE_STATIC);

  // add output index by config info and outputName
  std::unordered_map<std::string, int> tensorNameIndexMap;
  for (int i = 0; i < net->tensorName()->size(); ++i) {
    tensorNameIndexMap[net->tensorName()->Get(i)->str()] = i;
  }

  bool userSetOutput = false;
  for (auto& config : configs) {
    userSetOutput = userSetOutput || (!config.saveTensors.empty());
    for (const auto& name : config.saveTensors) {
      auto iter = tensorNameIndexMap.find(name);
      if (iter != tensorNameIndexMap.end()) {
        auto t = allTensors[iter->second].get();
        if (TensorUtils::getDescribe(t)->usage ==
            Tensor::InsideDescribe::NORMAL) {
          TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
        }
        scheduleInfo.outputTensor.insert(std::make_pair(
            net->tensorName()->GetAsString(iter->second)->c_str(), t));
      } else {
        MNN_PRINT("Bad outputname: %s\n", name.c_str());
      }
    }
  }

  LOG(INFO) << "net->outputName(): " << net->outputName();
  if (net->outputName()) {
    userSetOutput = userSetOutput || net->outputName()->size() >= 1;

    LOG(INFO) << "net->outputName()->size(): " << net->outputName()->size();

    for (int i = 0; i < net->outputName()->size(); ++i) {
      std::string name = net->outputName()->Get(i)->str();
      LOG(INFO) << "output name:" << name;
      auto iter = tensorNameIndexMap.find(name);

      if (iter != tensorNameIndexMap.end()) {
        auto t = allTensors[iter->second].get();
        if (TensorUtils::getDescribe(t)->usage ==
            Tensor::InsideDescribe::NORMAL) {
          TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
        }
        scheduleInfo.outputTensor.insert(std::make_pair(
            net->tensorName()->GetAsString(iter->second)->c_str(), t));
      }
    }
  }
  if (scheduleInfo.outputTensor.empty()) {
    userSetOutput = false;
  }

  // add input/output tensor to schedule's input/output
  for (int index = 0; index < allTensors.size(); index++) {
    auto t = allTensors[index].get();
    auto usage = TensorUtils::getDescribe(t)->usage;
    if (usage == Tensor::InsideDescribe::INPUT) {
      scheduleInfo.inputTensors.insert(
          std::make_pair(net->tensorName()->GetAsString(index)->c_str(), t));
    }
    if (usage == Tensor::InsideDescribe::OUTPUT && (!userSetOutput)) {
      scheduleInfo.outputTensor.insert(
          std::make_pair(net->tensorName()->GetAsString(index)->c_str(), t));
    }
  }

#ifndef MNN_BUILD_MINI
  for (auto iter = scheduleInfo.pipelineInfo.begin();
       iter != scheduleInfo.pipelineInfo.end();) {
    auto breakIndex = GeometryComputerUtils::buildConstantTensors(iter->second);
    if (breakIndex >= 0) {
      scheduleInfo.needInputContentForShape = true;
    }

#ifdef MNN_SEPERTE_SIZE
    LOG(INFO) << "MNN_SEPERTE_SIZE";
    if (breakIndex >= 0 && (breakIndex + 1) < iter->second.size()) {
      // Split oplist
      std::vector<Schedule::PipelineInfo> fuse;
      std::vector<Schedule::PipelineInfo> separate;
      fuse.insert(fuse.begin(), iter->second.begin(),
                  iter->second.begin() + breakIndex + 1);
      separate.insert(separate.begin(), iter->second.begin() + breakIndex + 1,
                      iter->second.end());
      oplists.clear();
      iter->second = std::move(separate);
      iter = scheduleInfo.pipelineInfo.insert(
          iter, std::make_pair(iter->first, fuse));
      iter++;
      iter++;
    } else {
      iter++;
    }
#else
    iter++;
#endif
  }
#endif
  return true;
}

}  // namespace tars

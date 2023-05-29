//
//  Session.cpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include <MNN/AutoTime.hpp>
#include <map>
#include <set>

#include "MNN_generated.h"
#include "core/AutoStorage.h"
#include "core/RuntimeFactory.hpp"
#include "core/Session.hpp"
#include "core/TensorUtils.hpp"
#include "glog/logging.h"
#include "utils/InitNet.hpp"

using namespace std;

namespace tars {
static void _createPipelineBackend(Schedule::PipelineInfo& iter,
                                   RuntimeInfo& runtime) {
  if (iter.first.cache.first != nullptr) {
    return;
  }
  // 总体上来说这些数据结构设计的太复杂了
  auto rt = runtime.first.find(iter.first.info.type)->second.get();
  auto cpuRuntime = runtime.second;
  bool specialUsage = false;
  if (iter.first.info.user != nullptr) {
    specialUsage = iter.first.info.user->flags > 0;
  }
  iter.first.cache.first.reset(rt->onCreate(iter.first.info.user));
  std::shared_ptr<Device> second;

  if (iter.first.cache.first->type() == MNN_FORWARD_CPU && (!specialUsage)) {
    iter.first.cache.second = iter.first.cache.first;
  } else {
    // Const Device shouldn't be used as default backend
    // The session may be schedule multi-thread but const backend is the same
    // We need create a new backend to do size compute / not support op compute
    BackendConfig defaultConfig;
    defaultConfig.flags = 4;
    iter.first.cache.second.reset(cpuRuntime->onCreate(&defaultConfig));
  }
}

Session::Session(Schedule::ScheduleInfo&& info, const ModeGroup& mode,
                 RuntimeInfo&& runtime) {
  mMode = mode;
  mRuntime = std::move(runtime);

  if (info.pipelineInfo.empty()) {
    mValid = false;
    return;
  }

  mInfo = std::move(info);
  LOG(INFO) << "pipeline list size: " << mInfo.pipelineInfo.size();

  for (auto& iter : mInfo.pipelineInfo) {
    // create a new pipeline
    _createPipelineBackend(iter, mRuntime);
    Pipeline::TuningAttr attr;
    attr.maxTuningNumber = mode.maxTuningNumber;
    attr.autoSetOpType = mode.backendMode == Interpreter::Session_Backend_Auto;
    auto rt = mRuntime.first.find(iter.first.info.type)->second.get();
    auto cpuRuntime = mRuntime.second;
    std::shared_ptr<Pipeline> newPipeline(new Pipeline(
        std::move(iter), mode.inputMode == Interpreter::Session_Input_Inside,
        mode.outputMode == Interpreter::Session_Output_User, attr, rt,
        cpuRuntime.get()));
    mPipelines.emplace_back(std::move(newPipeline));
  }

  mCallBackMode = mode.callBackMode;
}

Session::~Session() {
  for (auto& iter : mRuntime.first) {
    iter.second->mCancelled = true;
  }
  waitAsyncResize();
  mInfo.allTensors.clear();
  mPipelines.clear();
  mRuntime.first.clear();
  mRuntime.second = nullptr;
}
Schedule::PipelineInfo& Session::getPipelineInfo(int index) const {
  MNN_ASSERT(index >= 0);
  MNN_ASSERT(index < mPipelines.size());
  return mPipelines[index]->getPipelineInfo();
}

bool Session::loadCache(const void* buffer, size_t size) {
  for (auto iter : mRuntime.first) {
    auto res = iter.second->onSetCache(buffer, size);
    if (res) {
      return true;
    }
  }
  return false;
}
void Session::waitAsyncResize() {
  for (auto& iter : mRuntime.first) {
    iter.second->waitAsyncWork();
  }
}

bool Session::hasAsyncWork() {
  for (auto& iter : mRuntime.first) {
    auto res = iter.second->hasAsyncWork();
    if (res) {
      return true;
    }
  }
  return false;
}

std::pair<const void*, size_t> Session::getCache() {
  // Set cancelled for quickly ending
  for (auto& iter : mRuntime.first) {
    iter.second->mCancelled = true;
  }
  waitAsyncResize();

  for (auto iter : mRuntime.first) {
    auto res = iter.second->onGetCache();
    if (res.first != nullptr) {
      return res;
    }
  }
  return std::make_pair(nullptr, 0);
}

Status Session::run() const {
  if (mNeedResize) {
    MNN_ERROR("Can't run session because not resized\n");
    return Status::ERROR();
  }

  for (auto& iter : mPipelines) {
    auto error = iter->execute();
    if (NO_ERROR != error) {
      return error;
    }
  }
  return Status::OK();
}

Status Session::runWithCallBack(const TensorCallBackWithInfo& before,
                                const TensorCallBackWithInfo& end,
                                bool sync) const {
  if (mNeedResize) {
    MNN_ERROR("Can't run session because not resized\n");
    return Status::ERROR();
  }
  for (auto& iter : mPipelines) {
    auto error = iter->executeCallBack(before, end);
    if (NO_ERROR != error) {
      return error;
    }
  }
  return Status::OK();
}

void Session::_clearCache() {
  for (auto& t : mInfo.allTensors) {
    auto describe = TensorUtils::getDescribe(t.get());
    if (describe->usage == Tensor::InsideDescribe::TRAINABLE ||
        describe->usage == Tensor::InsideDescribe::CONSTANT) {
      continue;
    }
    describe->regions.clear();
  }
}

Status Session::resize() {
  bool firstMalloc = false;
  LOG(INFO) << "Session::resize: mNeedResize: " << mNeedResize;
  if (mNeedResize) {
    bool debug = mCallBackMode == Interpreter::Session_Debug;
    for (auto& iter : mPipelines) {
      auto error = iter->encode(debug);
    }
    mNeedResize = false;
    mNeedMalloc = true;
    firstMalloc = true;
  }
  if (mNeedMalloc) {
    // Set needResize = true for easy for judge in runSession when error
    mNeedResize = true;
    // Turn Pipeline to Command Buffer and Malloc resource
    // TODO: Separate Schedule and Malloc
    for (auto& iter : mPipelines) {
      auto error = iter->allocMemory(firstMalloc);
    }
    for (auto& iter : mRuntime.first) {
      iter.second->onGabageCollect(0);
    }
    mNeedMalloc = false;
    mNeedResize = false;
  }

  return Status::OK();
}

bool Session::getInfo(Interpreter::SessionInfoCode code, void* ptr) const {
  switch (code) {
    case Interpreter::MEMORY: {
      auto dst = (float*)ptr;
      float summer = mRuntime.second->onGetMemoryInMB();
      for (auto& r : mRuntime.first) {
        if (r.second.get() != mRuntime.second.get()) {
          summer += r.second->onGetMemoryInMB();
        }
      }
      *dst = summer;
      return true;
    } break;
    case Interpreter::BACKENDS: {
      int pos = 0;
      auto res = (int32_t*)ptr;
      for (auto& r : mPipelines) {
        auto type = r->getMainForwardType();
        res[pos++] = type;
      }
      return true;
    } break;
    case Interpreter::FLOPS: {
      float flo = 0.0f;
      for (auto& iter : mPipelines) {
        flo += iter->flops();
      }
      auto dst = (float*)ptr;
      *dst = flo;
      return true;
    } break;
    case Interpreter::RESIZE_STATUS: {
      auto dst = (int*)ptr;
      if (mNeedResize) {
        *dst = 2;
      } else if (mNeedMalloc) {
        *dst = 1;
      } else {
        *dst = 0;
      }
    } break;
    // TODO: Support other debug info
    default:
      break;
  }
  return false;
}

const Device* Session::getBackEnd(const Tensor* tensor) const {
  return TensorUtils::getDescribe(tensor)->backend;
}

Tensor* Session::getInput(const char* name) const {
  // MNN_ASSERT(!mInputs.empty());
  if (nullptr == name) {
    LOG(INFO) << "name is empty.";
    LOG(INFO) << "mInfo.inputTensors.size: " << mInfo.inputTensors.size();
    return mInfo.inputTensors.begin()->second;
  }
  auto iter = mInfo.inputTensors.find(name);
  if (iter == mInfo.inputTensors.end()) {
    MNN_PRINT("Error: can't find input: %s\n", name);
    return nullptr;
  }
  return iter->second;
}

Tensor* Session::getTensor(int index) const {
  return mInfo.allTensors[index].get();
}

Tensor* Session::getOutput(const char* name) const {
  MNN_ASSERT(!mInfo.outputTensor.empty());
  if (nullptr == name) {
    return mInfo.outputTensor.begin()->second;
  }

  auto iter = mInfo.outputTensor.find(name);
  if (iter == mInfo.outputTensor.end()) {
    MNN_PRINT("Error: can't find output: %s\n", name);
    return nullptr;
  }
  return iter->second;
}

const std::map<std::string, Tensor*>& Session::getInputAll() const {
  return mInfo.inputTensors;
}

const std::map<std::string, Tensor*>& Session::getOutputAll() const {
  return mInfo.outputTensor;
}

Status Session::updateToModel(Net* net) const {
  if (mNeedResize) {
    return Status::ERROR();
  }
  int opSize = net->oplists()->size();
  for (int i = 0; i < opSize; ++i) {
    auto op = net->oplists()->GetAs<Op>(i);
    if ((net->usage() == Usage_INFERENCE ||
         net->usage() == Usage_INFERENCE_STATIC) &&
        op->type() != OpType_Const) {
      continue;
    }
    if (net->usage() == Usage_TRAIN && op->type() != OpType_TrainableParam) {
      continue;
    }
    if (!op->outputIndexes() || op->outputIndexes()->size() != 1) {
      continue;
    }
    auto index = op->outputIndexes()->data()[0];
    auto blob = op->main_as_Blob();
    if (blob->dataType() != DataType_DT_FLOAT) {
      continue;
    }
    std::shared_ptr<Tensor> tensor = mInfo.allTensors[index];
    if (tensor->host<void>() == nullptr && tensor->deviceId() != 0) {
      tensor.reset(Tensor::createHostTensorFromDevice(tensor.get(), true));
      if (tensor.get() == nullptr) {
        MNN_ERROR("failed to copy trained param from device to host\n");
        return Status::ERROR();
      }
    }
    ::memcpy((void*)blob->float32s()->data(), tensor->host<float>(),
             tensor->size());
  }

  return Status::OK();
}

static void initTensors(std::vector<std::shared_ptr<Tensor>>& tensors,
                        const std::vector<std::shared_ptr<Tensor>>& tensorSrc) {
  LOG(INFO) << "tensor.size(): " << tensors.size();
  for (int i = 0; i < tensors.size(); ++i) {
    // Init all tensor except for const
    if (tensors[i].get() == nullptr) {
      tensors[i].reset(new Tensor);
      TensorUtils::getDescribe(tensors[i].get())->index = i;
    }
  }
  for (int i = 0; i < tensors.size(); ++i) {
    auto srcDes = TensorUtils::getDescribe(tensorSrc[i].get());
    if (srcDes->quantAttr != nullptr) {
      TensorUtils::getDescribe(tensors[i].get())
          ->quantAttr.reset(new QuantAttr);
      *TensorUtils::getDescribe(tensors[i].get())->quantAttr =
          *srcDes->quantAttr;
    }
    TensorUtils::copyShape(tensorSrc[i].get(), tensors[i].get(), true);
  }
}

Session* Session::clone(RuntimeInfo&& runtime,
                        std::shared_ptr<Schedule::ScheduleInfo> sharedConst) {
  // TODO: Currently only valid for Module api's onClone
  Schedule::ScheduleInfo scheduleInfo;
  scheduleInfo.defaultBackend = mInfo.defaultBackend;
  scheduleInfo.pipelineInfo.resize(1);
  Session::ModeGroup modes;
  scheduleInfo.defaultBackend = sharedConst->defaultBackend;
  scheduleInfo.allTensors = sharedConst->allTensors;
  initTensors(scheduleInfo.allTensors, mInfo.allTensors);
  MNN_ASSERT(1 == mPipelines.size());
  auto& srcPipelineInfo = mPipelines[0]->getPipelineInfo();
  auto& opCaches = srcPipelineInfo.second;
  auto& pipelineInfo = scheduleInfo.pipelineInfo[0];
  pipelineInfo.first.info = srcPipelineInfo.first.info;
  pipelineInfo.first.config = srcPipelineInfo.first.config;
  pipelineInfo.first.info.user = &pipelineInfo.first.config;
  auto& oplists = pipelineInfo.second;
  oplists.resize(opCaches.size());
  _createPipelineBackend(pipelineInfo, runtime);
  auto first = pipelineInfo.first.cache.first;
  auto second = pipelineInfo.first.cache.second;
  for (int i = 0; i < opCaches.size(); ++i) {
    auto& srcOpInfo = opCaches[i];
    auto& opInfo = oplists[i];
    opInfo.op = opCaches[i].op;
    opInfo.type = srcOpInfo.type;
    auto op = opInfo.op;
    if (nullptr != op->outputIndexes()) {
      auto data = op->outputIndexes()->data();
      for (int j = 0; j < op->outputIndexes()->size(); ++j) {
        opInfo.outputs.push_back(scheduleInfo.allTensors[data[j]].get());
      }
    }
    if (nullptr != op->inputIndexes()) {
      auto data = op->inputIndexes()->data();
      for (int j = 0; j < op->inputIndexes()->size(); ++j) {
        opInfo.inputs.push_back(scheduleInfo.allTensors[data[j]].get());
      }
    }
    for (int j = 0; j < opInfo.inputs.size(); ++j) {
      TensorUtils::getDescribe(opInfo.inputs[j])->usage =
          TensorUtils::getDescribe(srcOpInfo.inputs[j])->usage;
    }
    for (int j = 0; j < opInfo.outputs.size(); ++j) {
      TensorUtils::getDescribe(opInfo.outputs[j])->usage =
          TensorUtils::getDescribe(srcOpInfo.outputs[j])->usage;
    }
    // Clone cache
    for (auto& iter : srcOpInfo.executionCache) {
      Operation* copyExecution = nullptr;
      bool valid = false;
      if (first->type() == iter.second->backend()->type()) {
        valid = iter.second->onClone(first.get(), iter.first, &copyExecution);
      } else {
        valid = iter.second->onClone(second.get(), iter.first, &copyExecution);
      }
      if (valid) {
        std::shared_ptr<Operation> copyExeWrap(copyExecution);
        opInfo.executionCache.insert(std::make_pair(iter.first, copyExeWrap));
      }
    }
  }
  auto dst = new Session(std::move(scheduleInfo), mMode, std::move(runtime));
  return dst;
}

}  // namespace tars
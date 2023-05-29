#include <math.h>
#include <stdio.h>

#include <MNN/AutoTime.hpp>
#include <algorithm>
#include <mutex>
#include <vector>

#include "MNN_generated.h"
#include "core/AutoStorage.h"
#include "core/FileLoader.hpp"
#include "core/Interpreter.hpp"
#include "core/Pipeline.hpp"
#include "core/RuntimeFactory.hpp"
#include "core/Session.hpp"
#include "device/cpu/CPUDevice.h"
#include "glog/logging.h"

namespace tars {

struct Content {
  // Buffer
  AutoStorage<uint8_t> buffer;
  // the real DAG
  const Net* net = nullptr;
  // a session list
  std::vector<std::unique_ptr<Session>> sessions;
  // the tensor map
  std::map<Tensor*, const Session*> tensorMap;
  Session::ModeGroup modes;
  AutoStorage<uint8_t> cacheBuffer;
  std::string cacheFile;
  std::mutex lock;
  size_t lastCacheSize = 0;
  std::string bizCode;
  std::string uuid;
  bool mStaticShape = false;
  std::string externalFile;
#ifdef MNN_INTERNAL_ENABLED
  std::map<std::string, std::string> basicLogginData;
  std::map<const Session*, std::tuple<int, int>> sessionInfo;
#endif
};

static void writeCacheFile(const Content* net,
                           std::pair<const void*, size_t> buffer) {
  bool res = FileLoader::write(net->cacheFile.c_str(), buffer);
  if (!res) {
    MNN_ERROR("Write Cache File error!\n");
    return;
  }
}

static Content* loadModelFile(const char* file) {
  if (nullptr == file) {
    MNN_PRINT("NULL file for create interpreter\n");
    return nullptr;
  }
  std::unique_ptr<FileLoader> loader(new FileLoader(file));
  if (!loader->valid()) {
    MNN_PRINT("Create interpreter failed, open %s error\n", file);
    return nullptr;
  }
  bool result = loader->read();
  if (!result) {
    MNN_PRINT("Read file error\n");
    return nullptr;
  }
  if (loader->size() == 0) {
    MNN_PRINT("Create interpreter failed, %s is empty\n", file);
    return nullptr;
  }
  auto net = new Content;
  bool success = loader->merge(net->buffer);
  if (!success) {
    return nullptr;
  }
  loader.reset();
  return net;
}

Interpreter* Interpreter::createFromFile(const char* file) {
  Content* net = loadModelFile(file);
  if (nullptr == net) {
    return nullptr;
  }

  return createFromBufferInternal(net, true);
}

Interpreter* Interpreter::createFromBuffer(const void* buffer, size_t size) {
  if (buffer == nullptr || size == 0) {
    LOG(INFO) << "Buffer is null for create interpreter";
    return nullptr;
  }

  LOG(INFO) << "Model Buffer Size: " << size;
  auto net = new Content;
  net->buffer.reset((int)size);
  if (net->buffer.get() == nullptr) {
    LOG(INFO) << "Memory not enough!";
    return nullptr;
  }
  ::memcpy(net->buffer.get(), buffer, size);

  return createFromBufferInternal(net, true);
}

Interpreter* Interpreter::createFromBufferInternal(Content* net,
                                                   bool enforceAuth) {
  if (nullptr == net) {
    LOG(INFO) << "Buffer is null for create interpreter";
    return nullptr;
  }

  flatbuffers::Verifier verify((const uint8_t*)(net->buffer.get()),
                               net->buffer.size());
  if (false == VerifyNetBuffer(verify)) {
    LOG(INFO) << "Invalidate buffer to create interpreter";
    delete net;
    return nullptr;
  }

  net->net = GetNet(net->buffer.get());

  if (net->net->oplists() == nullptr) {
    LOG(INFO) << "Model has no oplist";
    delete net;
    return nullptr;
  }

  net->mStaticShape = net->net->usage() == Usage_INFERENCE_STATIC;
  int opSize = net->net->oplists()->size();
  LOG(INFO) << "Op size: " << opSize;
  for (int i = 0; i < opSize; ++i) {
    auto op = net->net->oplists()->GetAs<Op>(i);
    if (nullptr == op || nullptr == op->outputIndexes()) {
      LOG(INFO) << "Invalid Model, the " << i << " op is empty";
      delete net;
      return nullptr;
    }
  }
  return new Interpreter(net);
}

void Interpreter::setSessionHint(HintMode mode, int hint) {
  switch (mode) {
    case MAX_TUNING_NUMBER:
      mNet->modes.maxTuningNumber = hint;
      break;
    default:
      break;
  }
}

void Interpreter::setSessionMode(SessionMode mode) {
  if (mode == Session_Input_Inside || mode == Session_Input_User) {
    mNet->modes.inputMode = mode;
  } else if (mode == Session_Output_User || mode == Session_Output_Inside) {
    mNet->modes.outputMode = mode;
  } else if (mode == Session_Backend_Auto || mode == Session_Backend_Fix) {
    mNet->modes.backendMode = mode;
  } else if (mode == Session_Debug || mode == Session_Release) {
    mNet->modes.callBackMode = mode;
  } else if (mode == Session_Resize_Direct || mode == Session_Resize_Defer) {
    mNet->modes.resizeMode = mode;
  }
}

void Interpreter::setCacheFile(const char* cacheFile, size_t keySize) {
  if (nullptr == cacheFile || nullptr == mNet->buffer.get()) {
    MNN_ERROR("Empty cacheFile or the interpreter invalid\n");
    return;
  }
  mNet->cacheFile = std::string(cacheFile);
  std::unique_ptr<FileLoader> loader(new FileLoader(cacheFile));
  if (!loader->valid()) {
    MNN_ERROR("Load Cache file error.\n");
    return;
  }
  bool result = loader->read();
  if (!result) {
    MNN_ERROR("Load Cache file error.\n");
    return;
  }
  if (loader->size() == 0) {
    MNN_ERROR("Load Cache file error.\n");
    return;
  }
  bool success = loader->merge(mNet->cacheBuffer);
  if (!success) {
    MNN_ERROR("Alloc memory for Cache error.\n");
    return;
  }
}

void Interpreter::setExternalFile(const char* file, size_t flag) {
  mNet->externalFile = file;
}

Status Interpreter::updateCacheFile(Session* session, int flag) {
  std::lock_guard<std::mutex> _l(mNet->lock);

  // Backend_Auto and no Async work, then don't need updateCache
  if (mNet->modes.backendMode == Session_Backend_Auto &&
      !(session->hasAsyncWork())) {
    return Status::OK();
  }

  // Get cache and write to file
  auto buffer = session->getCache();

  // When current cacheSize bigger than previous, update
  if (buffer.first != nullptr && buffer.second > mNet->lastCacheSize) {
    MNN_PRINT("Update cache to %s, from size:%zu -> size:%zu\n",
              mNet->cacheFile.c_str(), mNet->lastCacheSize, buffer.second);
    writeCacheFile(mNet, buffer);
    mNet->lastCacheSize = buffer.second;
  }
  // Reset cache
  session->loadCache(nullptr, 0);
  return Status::OK();
}

Interpreter::Interpreter(Content* net) {
  MNN_ASSERT(nullptr != net);
  mNet = net;
  // Store bizcode and uuid because we need them even after `releaseModel` is
  // called.
  mNet->bizCode =
      std::string(mNet->net->bizCode() ? mNet->net->bizCode()->c_str() : "");
  mNet->uuid =
      std::string(mNet->net->mnn_uuid() ? mNet->net->mnn_uuid()->c_str() : "");
}

Interpreter::~Interpreter() {
  {
    // If the session is running, we must not delete session
    std::unique_lock<std::mutex> _l(mNet->lock);
    mNet->sessions.clear();
    mNet->tensorMap.clear();
  }
  delete mNet;
}

Session* Interpreter::createMultiPathSession(
    const std::vector<ScheduleConfig>& configs) {
  RuntimeInfo runtime = createRuntime(configs);
  runtime.second->setExternalFile(mNet->externalFile);
  if (runtime.first.empty()) {
    LOG(INFO) << "Runtime not valid for create session";
    return nullptr;
  }
  return createMultiPathSession(configs, std::move(runtime));
}

Session* Interpreter::createMultiPathSession(
    const std::vector<ScheduleConfig>& configs, const RuntimeInfo& runtime) {
  if (mNet->buffer.get() == nullptr) {
    LOG(INFO) << "The model buffer has been released. Can't create session";
    return nullptr;
  }

  if (runtime.first.empty()) {
    LOG(INFO) << "Runtime not valid for create session";
    return nullptr;
  }

  // 放置一个互斥锁，目测意义不大
  std::unique_lock<std::mutex> _l(mNet->lock);
  int cacheMode = 0;  // No cache
  Schedule::ScheduleInfo info;
  auto success = Schedule::schedule(info, mNet->net, configs, runtime);
  if (!success) {
    return nullptr;
  }

  LOG(INFO) << "mNet->mStaticShape: " << mNet->mStaticShape;
  if (mNet->mStaticShape) {
    for (auto& pipInfo : info.pipelineInfo) {
      pipInfo.first.needComputeGeometry = false;
      pipInfo.first.needComputeShape = false;
    }
  }

  RuntimeInfo rt = runtime;
  bool valid = false;
  if (mNet->cacheBuffer.get() != nullptr) {
    for (auto iter : rt.first) {
      valid = iter.second->onSetCache(mNet->cacheBuffer.get(),
                                      mNet->cacheBuffer.size());
      if (!valid) {
        iter.second->onSetCache(nullptr, 0);
      }
      if (valid) {
        break;
      }
    }

    if (valid) {
      mNet->lastCacheSize = mNet->cacheBuffer.size();
      cacheMode = cacheMode | 1;  // READ cache
    }
  }
  LOG(INFO) << "Create a new session.";
  auto newSession = std::unique_ptr<Session>(
      new Session(std::move(info), mNet->modes, std::move(rt)));

  if (!newSession->valid()) {
    MNN_PRINT("Invalide Session!!\n");
    return nullptr;
  }
  auto result = newSession.get();
  auto validForResize = info.validForResize;
  LOG(INFO) << "validForResize: " << validForResize;

  if (validForResize && mNet->modes.inputMode == Session_Input_Inside &&
      mNet->modes.resizeMode == Session_Resize_Direct) {
    LOG(INFO) << "session resize";
    result->resize();
  }

  if ((!mNet->cacheFile.empty()) && (!valid) &&
      mNet->modes.backendMode == Session_Backend_Fix) {
    // Try to save extra cache
    auto buffer = result->getCache();
    if (buffer.first != nullptr && buffer.second > 0) {
      MNN_PRINT("Write cache to %s, size = %zu\n", mNet->cacheFile.c_str(),
                buffer.second);
      writeCacheFile(mNet, buffer);
      mNet->lastCacheSize = buffer.second;
      // Write Cache
      cacheMode = cacheMode | 2;
    }
  }
  // Reset cache
  result->loadCache(nullptr, 0);

  // insert a session
  mNet->sessions.emplace_back(std::move(newSession));

  return result;
}

Session* Interpreter::createSession(const ScheduleConfig& config) {
  return createMultiPathSession({config});
}

Session* Interpreter::createSession(const ScheduleConfig& config,
                                    const RuntimeInfo& runtime) {
  return createMultiPathSession({config}, runtime);
}

bool Interpreter::releaseSession(Session* session) {
  std::unique_lock<std::mutex> _l(mNet->lock);
  for (auto iter = mNet->sessions.begin(); iter != mNet->sessions.end();
       iter++) {
    // TODO Delete tensormap
    for (auto tIter = mNet->tensorMap.begin();
         tIter != mNet->tensorMap.end();) {
      if (tIter->second == session) {
        tIter = mNet->tensorMap.erase(tIter);
        continue;
      }
      tIter++;
    }

    if ((*iter).get() == session) {
      mNet->sessions.erase(iter);
      return true;
    }
  }
  return false;
}

#ifdef MNN_INTERNAL_ENABLED
void Interpreter::logForRunSession(const Session* session, float timeInMs,
                                   const char* api) const {
  int backendType[MNN_FORWARD_ALL];
  session->getInfo(tars::Interpreter::BACKENDS, backendType);
  float flops = 0.0f;
  session->getInfo(tars::Interpreter::FLOPS, &flops);
  float memory = 0.0f;
  session->getInfo(tars::Interpreter::MEMORY, &memory);
  std::map<std::string, std::string> metrics = mNet->basicLogginData;
  metrics.emplace("UUID", mNet->uuid);
  metrics.emplace(
      "Device",
      std::to_string(
          backendType[0]));  // "Precision" is not logged here. Don't need it.
  metrics.emplace("Time", std::to_string(timeInMs));
  metrics.emplace("API", api);
  metrics.emplace("Flops", std::to_string(flops));
  metrics.emplace("Memory", std::to_string(memory));
  auto iter = mNet->sessionInfo.find(session);
  if (iter != mNet->sessionInfo.end()) {
    metrics.emplace("Precision", std::to_string(std::get<0>(iter->second)));
    metrics.emplace("Mode", std::to_string(std::get<1>(iter->second)));
  }
  logAsync(metrics);
}
#endif

Status Interpreter::runSession(Session* session) const {
#ifdef MNN_INTERNAL_ENABLED
  Timer timer;
#endif
  Status Status = session->run();

#ifdef MNN_INTERNAL_ENABLED
  if (shouldLog(FREQ_LOW)) {
    waitSessionFinish(session);
    float costTime = (float)timer.durationInUs() / (float)1000;
    logForRunSession(session, costTime, "Interpreter::runSession");
  }
#endif  // MNN_INTERNAL_ENABLED

  return Status;
}

Tensor* Interpreter::getSessionInput(const Session* session, const char* name) {
  if (session == nullptr) {
    return nullptr;
  }
  LOG(INFO) << "getSessionInput| name: " << name;
  std::unique_lock<std::mutex> ul(mNet->lock);
  auto tensor = session->getInput(name);
  mNet->tensorMap.insert(std::make_pair(tensor, session));
  return tensor;
}

Tensor* Interpreter::getSessionOutput(const Session* session,
                                      const char* name) {
  if (session == nullptr) {
    return nullptr;
  }
  std::unique_lock<std::mutex> _l(mNet->lock);
  auto tensor = session->getOutput(name);
  mNet->tensorMap.insert(std::make_pair(tensor, session));
  return tensor;
}

const std::map<std::string, Tensor*>& Interpreter::getSessionInputAll(
    const Session* session) const {
  std::unique_lock<std::mutex> _l(mNet->lock);
  auto& tensors = session->getInputAll();
  for (auto& iter : tensors) {
    mNet->tensorMap.insert(std::make_pair(iter.second, session));
  }
  return tensors;
}

const std::map<std::string, Tensor*>& Interpreter::getSessionOutputAll(
    const Session* session) const {
  std::unique_lock<std::mutex> _l(mNet->lock);
  auto& tensors = session->getOutputAll();
  for (auto& iter : tensors) {
    mNet->tensorMap.insert(std::make_pair(iter.second, session));
  }
  return tensors;
}
void Interpreter::resizeSession(Session* session) { resizeSession(session, 0); }

void Interpreter::resizeSession(Session* session, int needRelloc) {
  std::unique_lock<std::mutex> _l(mNet->lock);
  if (mNet->buffer.get() == nullptr) {
    MNN_ERROR("The model buffer has been released. Can't resize session\n");
    return;
  }
  if (1 == needRelloc) {
    session->setNeedMalloc(true);
  }
  session->resize();
}

Status Interpreter::runSessionWithCallBack(const Session* session,
                                           const TensorCallBack& before,
                                           const TensorCallBack& after,
                                           bool sync) const {
  auto beforeWrap = [&before](const std::vector<Tensor*>& tensors,
                              const OperatorInfo* info) {
    return before(tensors, info->name());
  };
  auto afterWrap = [&after](const std::vector<Tensor*>& tensors,
                            const OperatorInfo* info) {
    return after(tensors, info->name());
  };
  return runSessionWithCallBackInfo(session, beforeWrap, afterWrap, sync);
}

void Interpreter::waitSessionFinish(const Session* session) const {
  for (auto& t : mNet->tensorMap) {
    if (t.second == session) {
      if (TensorUtils::getDescribe(t.first)->usage !=
          Tensor::InsideDescribe::INPUT) {
        t.first->wait(Tensor::MAP_TENSOR_READ, true);
      }
    }
  }
}

Status Interpreter::runSessionWithCallBackInfo(
    const Session* session, const TensorCallBackWithInfo& before,
    const TensorCallBackWithInfo& callBack, bool sync) const {
#ifdef MNN_INTERNAL_ENABLED
  Timer timer;
#endif
  Status Status = session->runWithCallBack(before, callBack, sync);

#ifdef MNN_INTERNAL_ENABLED
  if (shouldLog(FREQ_LOW)) {
    waitSessionFinish(session);
    float costTime = (float)timer.durationInUs() / (float)1000;
    logForRunSession(session, costTime,
                     "Interpreter::runSessionWithCallBackInfo");
  }
#endif  // MNN_INTERNAL_ENABLED

  return Status;
}

const Device* Interpreter::getBackend(const Session* session,
                                      const Tensor* tensor) const {
  return session->getBackEnd(tensor);
}

void Interpreter::releaseModel() {
  std::unique_lock<std::mutex> _l(mNet->lock);
  for (auto& session : mNet->sessions) {
    session->waitAsyncResize();
  }
  if (mNet->buffer.get() != nullptr &&
      mNet->net->usage() != Usage_INFERENCE_STATIC) {
    mNet->buffer.release();
  }
  mNet->cacheBuffer.release();
}

void Interpreter::resizeTensor(Tensor* tensor, int batch, int channel,
                               int height, int width) {
  if (tensor->getDimensionType() == Tensor::TENSORFLOW) {
    resizeTensor(tensor, {batch, height, width, channel});
  } else {
    resizeTensor(tensor, {batch, channel, height, width});
  }
}

void Interpreter::resizeTensor(Tensor* tensor, const std::vector<int>& dims) {
  std::unique_lock<std::mutex> _l(mNet->lock);
  MNN_ASSERT(nullptr != tensor);
  bool dirty = false;
  if (tensor->buffer().dimensions != dims.size()) {
    dirty = true;
  } else {
    for (int i = 0; i < dims.size(); ++i) {
      if (tensor->buffer().dim[i].extent != dims[i]) {
        dirty = true;
        break;
      }
    }
  }

  if (!dirty) {
    return;
  }

  tensor->buffer().dimensions = (int)dims.size();
  for (int i = 0; i < dims.size(); ++i) {
    tensor->buffer().dim[i].extent = dims[i];
  }

  auto relatedSessionIter = mNet->tensorMap.find(tensor);
  MNN_ASSERT(relatedSessionIter != mNet->tensorMap.end());
  ((tars::Session*)relatedSessionIter->second)->setNeedResize();
}

const char* Interpreter::bizCode() const { return mNet->bizCode.c_str(); }

const char* Interpreter::uuid() const { return mNet->uuid.c_str(); }

std::pair<const void*, size_t> Interpreter::getModelBuffer() const {
  return std::make_pair(mNet->buffer.get(), mNet->buffer.size());
}
Status Interpreter::updateSessionToModel(Session* session) {
  std::unique_lock<std::mutex> _l(mNet->lock);
  if (mNet->buffer.get() == nullptr) {
    MNN_ERROR(
        "Can't updateSessionToModel because you called releaseModel before\n");
    return Status::ERROR();
  }
  return session->updateToModel((Net*)mNet->net);
}

const char* Interpreter::getModelVersion() const {
  if (mNet && mNet->net && mNet->net->extraInfo() &&
      mNet->net->extraInfo()->version()) {
    return mNet->net->extraInfo()->version()->c_str();
  }
  return "<2.0.0";
}

bool Interpreter::getSessionInfo(const Session* session, SessionInfoCode code,
                                 void* ptr) {
  std::unique_lock<std::mutex> _l(mNet->lock);
  if (nullptr == session || nullptr == ptr) {
    return false;
  }
  return session->getInfo(code, ptr);
}

static void _getDefaultBackend(RuntimeInfo& rt) {
  auto defaultType = MNN_FORWARD_CPU;
  if (rt.first.find(defaultType) != rt.first.end()) {
    rt.second = rt.first[defaultType];
  }
  if (rt.second == nullptr) {
    Device::Info info;
    info.type = defaultType;
    info.numThread = 1;
    rt.second.reset(RuntimeFactory::create(info));
  }
}

RuntimeInfo Interpreter::createRuntime(
    const std::vector<ScheduleConfig>& configs) {
  RuntimeInfo res;
  auto& mRuntimes = res.first;

  for (auto& config : configs) {
    Device::Info compute;
    compute.type = Schedule::getApprociateType(config);
    compute.numThread = config.numThread;
    compute.user = config.backendConfig;
    if (mRuntimes.find(compute.type) == mRuntimes.end()) {
      auto newBn = RuntimeFactory::create(compute);
      if (nullptr == newBn) {
        MNN_ERROR("Can't create Runtime: %s\n",
                  EnumNameForwardType((ForwardType)compute.type));
        continue;
      }
      mRuntimes[compute.type].reset(newBn);
    }
  }
  _getDefaultBackend(res);
  return res;
}

void Interpreter::destroy(Interpreter* net) {
  if (nullptr != net) {
    delete net;
  }
}

}  // namespace tars

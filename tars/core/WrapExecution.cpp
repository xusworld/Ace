//
//  WrapExecution.cpp
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>

#include "core/Concurrency.h"
#include "core/OpCommonUtils.hpp"
#include "core/TensorUtils.hpp"
#include "core/WrapExecution.hpp"
#include "device/cpu/CPUCast.hpp"
#include "device/cpu/compute/CommonOptFunction.h"

// #define LOG_VERBOSE

namespace tars {
bool WrapExecution::needWrap(const Tensor* input, Device* curBackend) {
  auto curType = curBackend ? curBackend->type() : MNN_FORWARD_CPU;
  if (curType == MNN_FORWARD_NN) {
    return false;
  }
  auto des = TensorUtils::getDescribe(input);
  auto bn = des->backend;
  MNNForwardType type = MNN_FORWARD_CPU;
  int pack = 4;
  int bytes = 4;
  if (nullptr != bn) {
    type = bn->type();
    if (type == MNN_FORWARD_CPU_EXTENSION) {
      auto core = static_cast<CPUDevice*>(bn)->functions();
      pack = core->pack;
      bytes = core->bytes;
    }
  }
  if (type == curType) {
    return false;
    ;
  }
  bool srcCpu = (type == MNN_FORWARD_CPU_EXTENSION || type == MNN_FORWARD_CPU);
  bool dstCpu =
      ((curType == MNN_FORWARD_CPU_EXTENSION) || (curType == MNN_FORWARD_CPU));
  if (srcCpu && dstCpu) {
    int curBytes = 4, curPack = 4;
    if (curBackend) {
      auto dstCore = static_cast<CPUDevice*>(curBackend)->functions();
      curBytes = dstCore->bytes;
      curPack = dstCore->pack;
    }
    if (curBytes == bytes) {
      if (curPack == pack || des->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
        return false;
      }
    }
  }
  return true;
}

class WrapCopyExecution : public Operation {
 public:
  WrapCopyExecution(Device* bn, Device* backup) : Operation(bn) {
    mBackupBackend = backup;
  }
  virtual ~WrapCopyExecution() {
    // Do nothing
  }
  virtual Status onResize(const std::vector<Tensor*>& inputs,
                          const std::vector<Tensor*>& outputs) override {
    auto inputBn = TensorUtils::getDescribe(inputs[0])->backend;
    auto outputBn = backend();
    auto inputForwardtype = MNN_FORWARD_CPU;
    if (nullptr != inputBn) {
      inputForwardtype = inputBn->type();
    }
    mMidCPUTensor = nullptr;
    if (inputForwardtype != MNN_FORWARD_CPU &&
        outputBn->type() != MNN_FORWARD_CPU) {
      // Use Mid Buffer
      mMidCPUTensor = WrapExecution::makeCopyTensor(inputs[0], mBackupBackend);
      auto res =
          mBackupBackend->onAcquireBuffer(mMidCPUTensor.get(), Device::DYNAMIC);
      if (!res) {
        return Status::ERROR();
      }
      mBackupBackend->onReleaseBuffer(mMidCPUTensor.get(), Device::DYNAMIC);
    }
    return Status::OK();
  }
  virtual Status onExecute(const std::vector<Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs) override {
    auto inputBn = TensorUtils::getDescribe(inputs[0])->backend;
    auto outputBn = backend();
    auto inputForwardtype = MNN_FORWARD_CPU;
    if (nullptr != mMidCPUTensor.get()) {
      inputBn->onCopyBuffer(inputs[0], mMidCPUTensor.get());
      outputBn->onCopyBuffer(mMidCPUTensor.get(), outputs[0]);
      return Status::OK();
    }
    if (outputBn->type() == MNN_FORWARD_CPU) {
      MNN_ASSERT(nullptr != inputBn);
      inputBn->onCopyBuffer(inputs[0], outputs[0]);
    } else {
      outputBn->onCopyBuffer(inputs[0], outputs[0]);
    }
    return Status::OK();
  }

 private:
  std::shared_ptr<Tensor> mMidCPUTensor;
  Device* mBackupBackend;
};
std::shared_ptr<Tensor> WrapExecution::makeCopyTensor(Tensor* t,
                                                      Device* targetBackend) {
  std::shared_ptr<Tensor> wrapTensor(new Tensor);
  TensorUtils::copyShape(t, wrapTensor.get(), true);
  wrapTensor->buffer().type = t->buffer().type;
  TensorUtils::adjustTensorForCompability(wrapTensor.get());
  TensorUtils::getDescribe(wrapTensor.get())->quantAttr =
      TensorUtils::getDescribe(t)->quantAttr;
  TensorUtils::getDescribe(wrapTensor.get())->backend = targetBackend;
  return wrapTensor;
}

std::pair<Operation*, std::shared_ptr<Tensor>> WrapExecution::makeCopyExecution(
    Device* backend, Device* backupBackend, Tensor* tensor,
    std::map<std::pair<Tensor*, Device*>, std::shared_ptr<Tensor>>& cache) {
  auto iter = cache.find(std::make_pair(tensor, backend));
  if (iter != cache.end()) {
    return std::make_pair((Operation*)nullptr, iter->second);
  }
  auto t = tensor;
  std::shared_ptr<Tensor> wrapTensor = makeCopyTensor(tensor, backend);
  cache.insert(std::make_pair(std::make_pair(t, backend), wrapTensor));
  Operation* copyExe = new WrapCopyExecution(backend, backupBackend);
  return std::make_pair(copyExe, wrapTensor);
}

std::shared_ptr<Tensor> WrapExecution::copyConstCache(
    Tensor* t, Device* curBackend,
    std::map<Tensor*, std::shared_ptr<Tensor>>& cache) {
  auto des = TensorUtils::getDescribe(t);
  if (curBackend->type() != MNN_FORWARD_CPU) {
    auto constCacheiter = cache.find(t);
    if (constCacheiter != cache.end()) {
      // The tensor has been copy by op before, just use it
      return constCacheiter->second;
    } else {
      // search or create const for new backend
      std::shared_ptr<Tensor> wrapTensor(new Tensor);
      TensorUtils::copyShape(t, wrapTensor.get(), true);
      wrapTensor->buffer().type = t->buffer().type;
      TensorUtils::adjustTensorForCompability(wrapTensor.get());
      TensorUtils::getDescribe(wrapTensor.get())->quantAttr =
          TensorUtils::getDescribe(t)->quantAttr;
      TensorUtils::getDescribe(wrapTensor.get())->usage =
          Tensor::InsideDescribe::CONSTANT;
      auto tempRes =
          curBackend->onAcquireBuffer(wrapTensor.get(), Device::STATIC);
      if (!tempRes) {
        return nullptr;
      }
      TensorUtils::getDescribe(wrapTensor.get())->backend = curBackend;
      curBackend->onCopyBuffer(t, wrapTensor.get());
      cache.insert(std::make_pair(t, wrapTensor));
      return wrapTensor;
    }
  }
  return nullptr;
}

}  // namespace tars

//
//  CPUDevice.hpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDevice_hpp
#define CPUDevice_hpp

#include <map>
#include <memory>

#include "MNN_generated.h"
#include "core/device.h"
#include "core/operation.h"

namespace tars {

class BufferAllocator;

class CPURuntime : public Runtime {
 public:
  friend class CPUDevice;
  CPURuntime(const Device::Info& info);
  virtual ~CPURuntime();
  int onGetRuntimeStatus(RuntimeStatus statusEnum) const override;
  virtual Device* onCreate(const BackendConfig* config) const override;
  virtual void onGabageCollect(int level) override;
  virtual float onGetMemoryInMB() override;
  virtual CompilerType onGetCompilerType() const override {
    return Compiler_Loop;
  }
  void onConcurrencyBegin() const;
  void onConcurrencyEnd() const;

 private:
  // allocator
  std::shared_ptr<BufferAllocator> mStaticAllocator;
  // threads number
  int mThreadNumber;
  mutable int mTaskIndex;
  BackendConfig::MemoryMode mMemory;
  BackendConfig::PowerMode mPower;
  BackendConfig::PrecisionMode mPrecision;

  // Device features
  // CPU features
  float mFlops = 0.0f;
  static Device* (*gExtraCreate)(const Runtime* runtime);
  size_t mFlags = 0;
};

struct CoreFunctions;

struct CoreInt8Functions;

class CPUResizeCache;

class CPUDevice : public Device {
 public:
  CPUDevice(const CPURuntime* runtime, BackendConfig::PrecisionMode precision,
            MNNForwardType type = MNN_FORWARD_CPU, size_t flags = 0);
  virtual ~CPUDevice();

  // Return sizeDivide, scheduleNumber aligned memory
  std::pair<int, int> multiThreadDivide(int size) const;

 public:
  virtual MemObj* onAcquire(const Tensor* nativeTensor,
                            StorageType storageType) override;
  virtual bool onClearBuffer() override;
  virtual void onCopyBuffer(const Tensor* srcTensor,
                            const Tensor* dstTensor) const override;

  virtual Operation* onCreate(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs,
                              const tars::Op* op) override;

  virtual void onExecuteBegin() const override;
  virtual void onExecuteEnd() const override;

  const CoreFunctions* functions() const { return mCoreFunctions; }
  // Return element size for Tensor, conside pack
  int getTensorSize(const Tensor* tensor, bool multiBytes = false) const;
  const CoreInt8Functions* int8Functions() const { return mInt8CoreFunctions; }

 public:
  class Creator {
   public:
    virtual Operation* onCreate(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                const tars::Op* op, Device* backend) const = 0;
  };

  static bool addCreator(OpType t, Creator* c);

  int threadNumber() const { return mRuntime->mThreadNumber; }

  BufferAllocator* getBufferAllocator() const {
    return mDynamicAllocator.get();
  }

  BackendConfig::MemoryMode memoryMode() const { return mRuntime->mMemory; }

  BackendConfig::PrecisionMode precisionMode() const { return mPrecisionMode; }

  CPUResizeCache* getCache() const { return mCache; }

  virtual const Runtime* getRuntime() override;

#ifdef MNN_USE_THREAD_POOL
  inline int taskIndex() const { return mRuntime->mTaskIndex; }
#endif
  static void initCreatorMap();
  static int getBytes(const Device* backend, const Tensor* output);
  static DataType getDataType(const Tensor* tensor);

 protected:
  MemObj* allocBuffer(int size, Tensor* dest, StorageType storageType);
  const CoreFunctions* mCoreFunctions;
  const CoreInt8Functions* mInt8CoreFunctions;

 private:
  // allocator or memory pool
  std::shared_ptr<BufferAllocator> mStaticAllocator;
  std::shared_ptr<BufferAllocator> mDynamicAllocator;
  // runtime
  CPURuntime* mRuntime;
  // config
  BackendConfig::PrecisionMode mPrecisionMode;
  static std::map<OpType, CPUDevice::Creator*>* gCreator;
  CPUResizeCache* mCache;
};

/** execution cast wrapper. insert tensor cast dynamic. */
class CastWrapExecution : public Operation {
 public:
  CastWrapExecution(Device* backend, DataType runT)
      : Operation(backend), mRunType(runT) {}
  virtual Status onExecute(const std::vector<Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs) override;

 private:
  DataType mRunType;
};

#define REGISTER_CPU_OP_CREATOR(name, opType) \
  void ___##name##__##opType##__() {          \
    static name _temp;                        \
    CPUDevice::addCreator(opType, &_temp);    \
  }

#ifdef MNN_SUPPORT_DEPRECATED_OP
#define REGISTER_CPU_OP_CREATOR_OLD(name, opType) \
  void ___##name##__##opType##__() {              \
    static name _temp;                            \
    CPUDevice::addCreator(opType, &_temp);        \
  }

#else
#define REGISTER_CPU_OP_CREATOR_OLD(name, opType) \
  void ___##name##__##opType##__() {}
#endif

}  // namespace tars

#endif /* CPUDevice_hpp */

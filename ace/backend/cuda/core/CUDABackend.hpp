//
//  CUDABackend.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CUDABackend_hpp
#define CUDABackend_hpp

#include <set>
#include <vector>

#include "ace_generated.h"
#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include "core/Backend.hpp"
#include "core/BufferAllocator.hpp"
#include "core/ConvolutionCommon.hpp"
#include "core/Macro.h"
namespace ace {
namespace CUDA {
class MNN_PUBLIC CUDARuntimeWrapper : public Runtime {
 public:
  CUDARuntimeWrapper(BackendConfig::PrecisionMode precision,
                     BackendConfig::PowerMode power);
  virtual ~CUDARuntimeWrapper();
  virtual Backend *onCreate(const BackendConfig *config) const override;
  virtual void onGabageCollect(int level) override;
  bool isCreateError() const { return mIsCreateError; }
  virtual CompilerType onGetCompilerType() const override {
    return Compiler_Loop;
  }
  virtual float onGetMemoryInMB() override;

 private:
  std::shared_ptr<BufferAllocator> mBufferPool;
  std::shared_ptr<CUDARuntime> mCUDARuntime;
  bool mIsCreateError{false};
};

class CUDABackend final : public Backend {
 public:
  CUDABackend(std::shared_ptr<BufferAllocator> st,
              std::shared_ptr<CUDARuntime> rt);
  ~CUDABackend();

  CUDARuntime *getCUDARuntime();
  virtual bool onAcquireBuffer(const Tensor *nativeTensor,
                               StorageType storageType) override;
  virtual bool onReleaseBuffer(const Tensor *nativeTensor,
                               StorageType storageType) override;
  virtual bool onClearBuffer() override;

  virtual Execution *onCreate(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs,
                              const ace::Op *op) override;
  virtual void onResizeBegin() override;
  virtual void onResizeEnd() override;

  virtual void onExecuteBegin() const override;
  virtual void onExecuteEnd() const override;

  virtual void onCopyBuffer(const Tensor *srcTensor,
                            const Tensor *dstTensor) const override;

  class Creator {
   public:
    virtual ~Creator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &output,
                                const ace::Op *op, Backend *backend) const = 0;
  };

  static bool addCreator(OpType t, Creator *c);

  BufferAllocator *getBufferPool() const { return mBufferPool.get(); }
  BufferAllocator *getStaticBufferPool() const {
    return mStaticBufferPool.get();
  }
  virtual std::pair<float, bool> onMeasure(const std::vector<Tensor *> &inputs,
                                           const std::vector<Tensor *> &outputs,
                                           const ace::Op *op) override;
  static size_t realSize(const Tensor *tensor);

 private:
  std::shared_ptr<BufferAllocator> mBufferPool;
  std::shared_ptr<BufferAllocator> mStaticBufferPool;
  std::shared_ptr<CUDARuntime> mCUDARuntime;
};

template <class T>
class CUDACreatorRegister {
 public:
  CUDACreatorRegister(OpType type) {
    T *t = new T;
    CUDABackend::addCreator(type, t);
  }
  ~CUDACreatorRegister() = default;
};

template <typename T>
class TypedCreator : public CUDABackend::Creator {
 public:
  virtual ~TypedCreator() = default;
  virtual Execution *onCreate(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs,
                              const ace::Op *op,
                              Backend *backend) const override {
    return new T(inputs, op, backend);
  }
};

}  // namespace CUDA
}  // namespace ace
#endif /* CUDABackend_hpp */
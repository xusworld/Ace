//
//  AVX2Backend.hpp
//  MNN
//
//  Created by MNN on 2021/05/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef AVX2Backend_hpp
#define AVX2Backend_hpp

#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "device/cpu/CPUDevice.h"

namespace tars {
class AVX2Backend : public CPUDevice {
 public:
  virtual ~AVX2Backend();
  AVX2Backend(const CPURuntime* runtime, size_t flags);
  virtual Operation* onCreate(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs,
                              const tars::Op* op) override;
  virtual Device::MemObj* onAcquire(const Tensor* nativeTensor,
                                    StorageType storageType) override;

  virtual void onCopyBuffer(const Tensor* srcTensor,
                            const Tensor* dstTensor) const override;

  int numberThread() const { return threadNumber(); }
  static bool isValid();
};

}  // namespace tars

#endif /* AVX2Backend_hpp */

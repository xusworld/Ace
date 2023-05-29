//
//  Wrapoperation.h
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef WrapExecution_hpp
#define WrapExecution_hpp

#include <stdio.h>

#include <memory>

#include "core/Macro.h"
#include "core/device.h"
#include "core/operation.h"
#include "device/cpu/CPUDevice.h"
#include "device/cpu/compute/Int8FunctionsOpt.h"

namespace tars {

/** execution wrapper. hiding cross-backend tensor converting. */
class MNN_PUBLIC WrapExecution {
 public:
  static bool needWrap(const Tensor* input, Device* current);
  static std::shared_ptr<Tensor> copyConstCache(
      Tensor* tensor, Device* curBackend,
      std::map<Tensor*, std::shared_ptr<Tensor>>& cache);
  static std::shared_ptr<Tensor> makeCopyTensor(Tensor* tensor,
                                                Device* targetBackend);
  static std::pair<Operation*, std::shared_ptr<Tensor>> makeCopyExecution(
      Device* backend, Device* backupBackend, Tensor* tensor,
      std::map<std::pair<Tensor*, Device*>, std::shared_ptr<Tensor>>& cache);
};

}  // namespace tars

#endif /* WrapExecution_hpp */

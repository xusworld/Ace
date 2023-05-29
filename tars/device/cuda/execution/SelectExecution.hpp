//
//  Selectoperation.h
//  MNN
//
//  Created by MNN on 2021/12/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SelectExecution_hpp
#define SelectExecution_hpp

#include <vector>

#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {

class SelectExecution : public Operation {
 public:
  SelectExecution(Device *backend);
  virtual ~SelectExecution() = default;

  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
};

}  // namespace cuda
}  // namespace tars
#endif /* SelectExecution_hpp */

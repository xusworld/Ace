//
//  Rangeoperation.h
//  MNN
//
//  Created by MNN on 2022/04/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef RangeExecution_hpp
#define RangeExecution_hpp

#include <vector>

#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {
class RangeExecution : public Operation {
 public:
  RangeExecution(Device *backend);
  virtual ~RangeExecution() = default;

  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
};

}  // namespace cuda
}  // namespace tars
#endif /* SelectExecution_hpp */

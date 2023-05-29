//
//  CPUSvd.hpp
//  MNN
//
//  Created by MNN on 2022/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSvd_hpp
#define CPUSvd_hpp

#include "core/operation.h"

namespace tars {
class CPUSvd : public Operation {
 public:
  CPUSvd(Device *backend) : Operation(backend) {
    // Do nothing
  }
  virtual ~CPUSvd() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  int mRow, mCol;
  std::shared_ptr<Tensor> mAt;
};

}  // namespace tars

#endif /* CPUSvd.hpp */

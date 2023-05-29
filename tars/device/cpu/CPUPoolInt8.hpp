//
//  CPUPoolInt8.hpp
//  MNN
//
//  Created by MNN on 2019/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUPoolInt8_hpp
#define CPUPoolInt8_hpp

#include "device/cpu/CPUDevice.h"

namespace tars {

class CPUPoolInt8 : public Operation {
 public:
  CPUPoolInt8(Device *backend, const Pool *parameter);
  virtual ~CPUPoolInt8() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  const Pool *mParameter;
  std::function<void(const Tensor *src, Tensor *dst)> mThreadFunction;
  // nhwc buffer
  std::shared_ptr<Tensor> mInputTemp;
  std::shared_ptr<Tensor> mOutputTemp;
};

}  // namespace tars

#endif /* CPUPoolInt8_hpp */

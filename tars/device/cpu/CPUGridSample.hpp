//
//  CPUGridSample.hpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUGridSample_hpp
#define CPUGridSample_hpp

#include "MNN_generated.h"
#include "core/operation.h"

namespace tars {
class CPUGridSample : public Operation {
 public:
  CPUGridSample(Device *b, SampleMode mode, BorderMode paddingMode,
                bool alignCorners);
  virtual ~CPUGridSample() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  SampleMode mMode;
  BorderMode mPaddingMode;
  bool mAlignCorners;
  std::shared_ptr<Tensor> mTempCordBuffer;
};

}  // namespace tars

#endif /* CPUGridSample_hpp */

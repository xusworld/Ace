//
//  CPUGridSample.hpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUGridSample_hpp
#define CPUGridSample_hpp

#include "ace_generated.h"
#include "core/Execution.hpp"

namespace ace {
class CPUGridSample : public Execution {
 public:
  CPUGridSample(Backend *b, SampleMode mode, BorderMode paddingMode,
                bool alignCorners);
  virtual ~CPUGridSample() = default;
  virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

 private:
  SampleMode mMode;
  BorderMode mPaddingMode;
  bool mAlignCorners;
  std::shared_ptr<Tensor> mTempCordBuffer;
};

}  // namespace ace

#endif /* CPUGridSample_hpp */

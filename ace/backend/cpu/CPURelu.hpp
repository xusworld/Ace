//
//  CPURelu.hpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPURelu_hpp
#define CPURelu_hpp

#include "core/AutoStorage.h"
#include "core/Execution.hpp"

namespace ace {
class CPURelu : public Execution {
 public:
  CPURelu(Backend *b, float slope);
  virtual ~CPURelu() = default;
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;
  virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;

 private:
  AutoStorage<uint8_t> mSlope;
  AutoStorage<uint8_t> mCacheSrc;
  AutoStorage<uint8_t> mCacheDst;
  int mRealSize;
};

class CPUPRelu : public Execution {
 public:
  CPUPRelu(Backend *b, const Op *op);
  virtual ~CPUPRelu();
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

 private:
  Tensor mSlope;
};

class CPURelu6 : public Execution {
 public:
  CPURelu6(float maxV, float minV, Backend *b) : Execution(b) {
    mParam = {1.0f, 0.0f, minV, maxV};
  }
  virtual ~CPURelu6() = default;
  virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

 private:
  std::vector<float> mParam;
  AutoStorage<uint8_t> mCacheSrc;
  AutoStorage<uint8_t> mCacheDst;
  int mRealSize;
};

}  // namespace ace

#endif /* CPURelu_hpp */

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
#include "core/operation.h"

namespace tars {
class CPURelu : public Operation {
 public:
  CPURelu(Device *b, float slope);
  virtual ~CPURelu() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;

 private:
  AutoStorage<uint8_t> mSlope;
  AutoStorage<uint8_t> mCacheSrc;
  AutoStorage<uint8_t> mCacheDst;
  int mRealSize;
};

class CPUPRelu : public Operation {
 public:
  CPUPRelu(Device *b, const Op *op);
  virtual ~CPUPRelu();
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  Tensor mSlope;
};

class CPURelu6 : public Operation {
 public:
  CPURelu6(float maxV, float minV, Device *b) : Operation(b) {
    mParam = {1.0f, 0.0f, minV, maxV};
  }
  virtual ~CPURelu6() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  std::vector<float> mParam;
  AutoStorage<uint8_t> mCacheSrc;
  AutoStorage<uint8_t> mCacheDst;
  int mRealSize;
};

}  // namespace tars

#endif /* CPURelu_hpp */

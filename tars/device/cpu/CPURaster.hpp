//
//  CPURaster.hpp
//  MNN
//
//  Created by MNN on b'2020/04/02'.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef CPURaster_hpp
#define CPURaster_hpp
#include <map>
#include <set>

#include "CPUDevice.h"
#include "core/TensorUtils.hpp"
namespace tars {
class CPURaster : public Operation {
 public:
  CPURaster(Device *bn) : Operation(bn) {
    // Do nothing
  }
  virtual ~CPURaster() {
    // Do nothing
  }

  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  void executeFaster(const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs) const;
  void tensorConvert(Tensor *input, Tensor *output, int bytes);

 private:
  std::map<Tensor *, Tensor *> mTempInput;
  std::vector<std::pair<void *, Tensor::InsideDescribe::Region *>>
      mTempInputCopy;
  std::vector<std::pair<void *, Tensor::InsideDescribe::Region>> mFastBlit;
  std::shared_ptr<Tensor> mTempOutput;
  void *mOutputPtr;
  bool mNeedZero = false;
  bool mFast = false;
  int mSingleConvert = 0;
  std::vector<std::shared_ptr<Tensor::InsideDescribe::Region>> mCacheRegions;
  int32_t mZeroPoint = 0;
};
}  // namespace tars
#endif

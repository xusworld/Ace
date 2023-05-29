//
//  CPUMatMul.hpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUMATMUL_HPP
#define CPUMATMUL_HPP

#include <functional>

#include "core/operation.h"
#include "device/cpu/compute/StrassenMatmulComputor.hpp"

namespace tars {

class CPUMatMul : public Operation {
 public:
  CPUMatMul(Device *backend, bool transposeA, bool transposeB, bool transposeC,
            bool multiThread);
  virtual ~CPUMatMul() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  void execute(const float *APtr, const float *BPtr, float *CPtr,
               const float *BiasPtr);

 private:
  void _scheduleForVec(int e, int l, int h);
  void _scheduleForVecE(int e, int l, int h);
  bool mTransposeA;
  bool mTransposeB;
  bool mTransposeC;
  bool mSupportMultiThread = false;
  std::vector<std::pair<
      std::function<void(int, const float *, const float *, const float *)>,
      int>>
      mPreFunctions;
  std::vector<std::pair<std::function<void(int, const float *, const float *,
                                           const float *, float *)>,
                        int>>
      mPostFunctions;
  std::shared_ptr<StrassenMatrixComputor> mComputer;
  bool mStrassenUseBiasDirectly = false;
};
}  // namespace tars

#endif  // CPUMATMUL_HPP

//
//  CPUInterp.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUInterp_hpp
#define CPUInterp_hpp

#include "device/cpu/CPUResize.hpp"

namespace tars {

class CPUInterp : public CPUResizeCommon {
 public:
  CPUInterp(Device *backend, int resizeType, float widthScale = 0.f,
            float heightScale = 0.f, float widthOffset = 0.f,
            float heightOffset = 0.f);
  virtual ~CPUInterp();
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;

 private:
  Tensor mWidthPosition;
  Tensor mWidthFactor;
  Tensor mHeightPosition;
  Tensor mHeightFactor;
  Tensor mLineBuffer;
  float mWidthScale;
  float mHeightScale;
  float mWidthOffset;
  float mHeightOffset;
  int mResizeType;  // 1:near 2: bilinear 3: cubic 4: nearest_round
  bool mInit = false;
};

}  // namespace tars

#endif /* CPUInterp_hpp */

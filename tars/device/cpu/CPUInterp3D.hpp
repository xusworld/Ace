//
//  CPUInterp.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUInterp3D_hpp
#define CPUInterp3D_hpp

#include "device/cpu/CPUResize.hpp"

namespace tars {

class CPUInterp3D : public CPUResizeCommon {
 public:
  CPUInterp3D(Device *backend, int resizeType, float widthScale = 0.f,
              float heightScale = 0.f, float depthScale = 0.f,
              float widthOffset = 0.f, float heightOffset = 0.f,
              float depthOffset = 0.f);
  virtual ~CPUInterp3D();
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;

 private:
  Tensor mWidthPosition;
  Tensor mWidthFactor;
  Tensor mHeightPosition;
  Tensor mHeightFactor;
  Tensor mDepthPosition;
  Tensor mDepthFactor;
  Tensor mLineBuffer;
  float mWidthScale;
  float mHeightScale;
  float mDepthScale;
  float mWidthOffset;
  float mHeightOffset;
  float mDepthOffset;
  int mResizeType;  // 1:near 2: bilinear 3: cubic 4: nearest_round
  bool mInit = false;
};

}  // namespace tars

#endif /* CPUInterp_hpp */

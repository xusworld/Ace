//
//  CPUROIAlign.hpp
//  MNN
//
//  Created by MNN on 2021/11/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUROIAlign_hpp
#define CPUROIAlign_hpp

#include "MNN_generated.h"
#include "core/operation.h"

namespace tars {
class CPUROIAlign : public Operation {
 public:
  CPUROIAlign(Device *backend, int pooledWidth, int pooledHeight,
              int samplingRatio, float spatialScale, bool aligned,
              PoolType poolType);
  virtual ~CPUROIAlign() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  Status preCalcBilinearInterpolate(int height, int width, int pooledHeight,
                                    int pooledWidth, float roiStartH,
                                    float roiStartW, float binSizeH,
                                    float binSizeW, int samplingRatioH,
                                    int samplingRatioW,
                                    std::vector<std::vector<int>> &vecPos,
                                    std::vector<std::vector<float>> &vecArea);

 private:
  int mPooledWidth;
  int mPooledHeight;
  int mSamplingRatio;
  float mSpatialScale;
  bool mAligned;
  PoolType mPoolType;

  Tensor mROI;
};

}  // namespace tars

#endif /* CPUROIAlign_hpp */
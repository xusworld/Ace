//
//  CPUROIPooling.hpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUROIPooling_hpp
#define CPUROIPooling_hpp

#include "core/operation.h"

namespace tars {

class CPUROIPooling : public Operation {
 public:
  CPUROIPooling(Device *backend, int pooledWidth, int pooledHeight,
                float spatialScale);
  virtual ~CPUROIPooling() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  int mPooledWidth;
  int mPooledHeight;
  float mSpatialScale;

  Tensor mROI;
};

}  // namespace tars

#endif /* CPUROIPooling_hpp */

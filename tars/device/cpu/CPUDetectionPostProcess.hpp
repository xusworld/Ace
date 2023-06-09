//
//  CPUDetectionPostProcess.hpp
//  MNN
//
//  Created by MNN on 2019/10/29.
//  Copyright © 2018, Alibaba Group Holding Limited

#ifndef CPUDetectionPostProcess_hpp
#define CPUDetectionPostProcess_hpp

#include "MNN_generated.h"
#include "core/operation.h"

namespace tars {

/*
 DetectionPostProcess
 This implementation reference from tflite detection_postprocess
 you can get ssd model from here:
 https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md
 */

struct BoxCornerEncoding {
  float ymin;
  float xmin;
  float ymax;
  float xmax;
};

struct CenterSizeEncoding {
  float y;
  float x;
  float h;
  float w;
};

class CPUDetectionPostProcess : public Operation {
 public:
  CPUDetectionPostProcess(Device *bn, const tars::Op *op);
  virtual ~CPUDetectionPostProcess() = default;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  DetectionPostProcessParamT mParam;

  std::shared_ptr<Tensor> mDecodedBoxes;
};

}  // namespace tars

#endif /* CPUDetectionPostProcess_hpp */

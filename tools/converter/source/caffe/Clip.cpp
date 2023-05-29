//
//  Clip.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Clip : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Clip() {}
  virtual ~Clip() {}
  virtual tars::OpType opType() { return tars::OpType_ReLU6; }
  virtual tars::OpParameter type() { return tars::OpParameter_Relu6; }
};

void Clip::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
               const caffe::LayerParameter& weight) {
  auto relu6 = new tars::Relu6T;
  relu6->minValue = parameters.clip_param().min();
  relu6->maxValue = parameters.clip_param().max();
  dstOp->main.value = relu6;
}

static OpConverterRegister<Clip> a(Clip);

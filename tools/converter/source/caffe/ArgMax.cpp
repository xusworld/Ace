//
//  ArgMax.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class ArgMax : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  ArgMax() {}
  virtual ~ArgMax() {}
  virtual tars::OpType opType() { return tars::OpType_ArgMax; }
  virtual tars::OpParameter type() { return tars::OpParameter_ArgMax; }
};

void ArgMax::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                 const caffe::LayerParameter& weight) {
  auto axisT = new tars::ArgMaxT;
  dstOp->main.value = axisT;
  auto& c = parameters.argmax_param();
  // in caffe, axis may not exist, we set it to 10000 to indicate this situation
  axisT->axis = 10000;
  if (c.has_axis()) {
    axisT->axis = c.axis();
  }
  axisT->outMaxVal = c.out_max_val();
  axisT->topK = c.top_k();
  axisT->softmaxThreshold = c.softmax_threshold();
}
static OpConverterRegister<ArgMax> a("ArgMax");

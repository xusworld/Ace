//
//  Softmax.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Softmax : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Softmax() {}
  virtual ~Softmax() {}
  virtual tars::OpType opType() { return tars::OpType_Softmax; }
  virtual tars::OpParameter type() { return tars::OpParameter_Axis; }
};

void Softmax::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                  const caffe::LayerParameter& weight) {
  auto axisT = new tars::AxisT;
  auto c = parameters.softmax_param();
  if (c.has_axis()) {
    axisT->axis = c.axis();
  } else {
    axisT->axis = 1;
  }
  dstOp->main.value = axisT;
}
static OpConverterRegister<Softmax> a("Softmax");

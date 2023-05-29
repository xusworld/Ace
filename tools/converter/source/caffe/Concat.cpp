//
//  Concat.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Concat : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Concat() {}
  virtual ~Concat() {}
  virtual tars::OpType opType() { return tars::OpType_Concat; }
  virtual tars::OpParameter type() { return tars::OpParameter_Axis; }
};

void Concat::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                 const caffe::LayerParameter& weight) {
  auto axisT = new tars::AxisT;
  dstOp->main.value = axisT;
  auto& c = parameters.concat_param();
  if (c.has_axis()) {
    axisT->axis = c.axis();
  } else {
    axisT->axis = 1;
  }
}
static OpConverterRegister<Concat> a("Concat");

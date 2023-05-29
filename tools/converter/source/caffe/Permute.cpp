//
//  Permute.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Permute : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Permute() {}
  virtual ~Permute() {}
  virtual tars::OpType opType() { return tars::OpType_Permute; }
  virtual tars::OpParameter type() { return tars::OpParameter_Permute; }
};

void Permute::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                  const caffe::LayerParameter& weight) {
  const ::caffe::PermuteParameter& par = parameters.permute_param();

  auto permute = new tars::PermuteT;
  dstOp->main.value = permute;

  for (int i = 0; i < par.order_size(); ++i) {
    permute->dims.push_back(par.order(i));
  }
}
static OpConverterRegister<Permute> __a("Permute");

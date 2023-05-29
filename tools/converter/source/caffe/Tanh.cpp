//
//  Tanh.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Tanh : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Tanh() {}
  virtual ~Tanh() {}
  virtual tars::OpType opType() { return tars::OpType_TanH; }
  virtual tars::OpParameter type() { return tars::OpParameter_NONE; }
};

void Tanh::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
               const caffe::LayerParameter& weight) {
  dstOp->main.value = nullptr;
}
static OpConverterRegister<Tanh> a("TanH");

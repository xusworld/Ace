//
//  Sigmoid.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Sigmoid : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Sigmoid() {}
  virtual ~Sigmoid() {}
  virtual tars::OpType opType() { return tars::OpType_Sigmoid; }
  virtual tars::OpParameter type() { return tars::OpParameter_NONE; }
};

void Sigmoid::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                  const caffe::LayerParameter& weight) {
  dstOp->main.value = nullptr;
}

static OpConverterRegister<Sigmoid> a("Sigmoid");

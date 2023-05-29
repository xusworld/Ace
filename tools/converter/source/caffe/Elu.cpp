//
//  Elu.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include "logkit.h"

class Elu : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Elu() {}
  virtual ~Elu() {}
  virtual tars::OpType opType() { return tars::OpType_ELU; }
  virtual tars::OpParameter type() { return tars::OpParameter_ELU; }
};

void Elu::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
              const caffe::LayerParameter& weight) {
  auto elu = new tars::ELUT;
  auto param = parameters.elu_param();
  elu->alpha = param.alpha();
  dstOp->main.value = elu;
}

static OpConverterRegister<Elu> a("ELU");

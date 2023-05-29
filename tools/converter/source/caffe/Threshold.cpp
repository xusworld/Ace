//
//  Threshold.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Threshold : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Threshold() {}
  virtual ~Threshold() {}
  virtual tars::OpType opType() { return tars::OpType_Threshold; }
  virtual tars::OpParameter type() { return tars::OpParameter_ELU; }
};

void Threshold::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                    const caffe::LayerParameter& weight) {
  auto threshold = parameters.threshold_param().threshold();
  auto parameter = new tars::ELUT;
  parameter->alpha = threshold;
  dstOp->main.value = parameter;
}

static OpConverterRegister<Threshold> ____a("Threshold");

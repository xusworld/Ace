//
//  Eltwise.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include "logkit.h"

class EltWise : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  EltWise() {}
  virtual ~EltWise() {}
  virtual tars::OpType opType() { return tars::OpType_Eltwise; }
  virtual tars::OpParameter type() { return tars::OpParameter_Eltwise; }
};

void EltWise::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                  const caffe::LayerParameter& weight) {
  auto elt = new tars::EltwiseT;
  dstOp->main.value = elt;
  auto& caffeParam = parameters.eltwise_param();
  switch (caffeParam.operation()) {
    case caffe::EltwiseParameter_EltwiseOp_MAX:
      elt->type = tars::EltwiseType_MAXIMUM;
      break;
    case caffe::EltwiseParameter_EltwiseOp_SUM:
      elt->type = tars::EltwiseType_SUM;
      break;
    case caffe::EltwiseParameter_EltwiseOp_PROD:
      elt->type = tars::EltwiseType_PROD;
      break;

    default:
      break;
  }

  const int coffSize = caffeParam.coeff_size();
  elt->coeff.resize(coffSize);
  for (int i = 0; i < coffSize; ++i) {
    elt->coeff[i] = caffeParam.coeff(i);
  }

  if (coffSize == 2) {
    if (elt->type == tars::EltwiseType_SUM &&
        (elt->coeff[0] == 1.0f && elt->coeff[1] == -1.0f)) {
      elt->type = tars::EltwiseType_SUB;
      elt->coeff.resize(0);
    } else if (elt->type == tars::EltwiseType_SUB &&
               (elt->coeff[0] == 1.0f && elt->coeff[1] == -1.0f)) {
      elt->type = tars::EltwiseType_SUM;
      elt->coeff.resize(0);
    }
  }
}
static OpConverterRegister<EltWise> a("Eltwise");

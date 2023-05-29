//
//  BNLL.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class BNLL : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  BNLL() {}
  virtual ~BNLL() {}
  virtual tars::OpType opType() { return tars::OpType_UnaryOp; }
  virtual tars::OpParameter type() { return tars::OpParameter_UnaryOp; }
};

void BNLL::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
               const caffe::LayerParameter& weight) {
  auto parameter = new tars::UnaryOpT;

  parameter->T = tars::DataType_DT_FLOAT;

  parameter->opType = tars::UnaryOpOperation_BNLL;

  dstOp->main.value = parameter;
}

static OpConverterRegister<BNLL> ____a("BNLL");

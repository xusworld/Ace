//
//  UnaryOp.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class UnaryOp : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  UnaryOp() {}
  virtual ~UnaryOp() {}
  virtual tars::OpType opType() { return tars::OpType_UnaryOp; }
  virtual tars::OpParameter type() { return tars::OpParameter_UnaryOp; }
};

void UnaryOp::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                  const caffe::LayerParameter& weight) {
  auto parameter = new tars::UnaryOpT;

  parameter->T = tars::DataType_DT_FLOAT;

  parameter->opType = tars::UnaryOpOperation_ABS;

  dstOp->main.value = parameter;
}

static OpConverterRegister<UnaryOp> ____a("AbsVal");

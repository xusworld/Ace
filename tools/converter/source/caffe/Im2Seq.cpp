//
//  Im2Seq.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Im2Seq : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Im2Seq() {}
  virtual ~Im2Seq() {}
  virtual tars::OpType opType() { return tars::OpType_Im2Seq; }
  virtual tars::OpParameter type() { return tars::OpParameter_NONE; }
};

void Im2Seq::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                 const caffe::LayerParameter& weight) {}
static OpConverterRegister<Im2Seq> a("Im2seq");

class Seq2Out : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight) {}
  Seq2Out() {}
  virtual ~Seq2Out() {}
  virtual tars::OpType opType() { return tars::OpType_Seq2Out; }
  virtual tars::OpParameter type() { return tars::OpParameter_NONE; }
};
static OpConverterRegister<Seq2Out> b("Seq2out");

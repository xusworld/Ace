//
//  Relu.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include "logkit.h"

class Relu : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Relu() {}
  virtual ~Relu() {}
  virtual tars::OpType opType() { return tars::OpType_ReLU; }
  virtual tars::OpParameter type() { return tars::OpParameter_Relu; }
};

class Relu6 : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Relu6() {}
  virtual ~Relu6() {}
  virtual tars::OpType opType() { return tars::OpType_ReLU6; }
  virtual tars::OpParameter type() { return tars::OpParameter_Relu6; }
};

void Relu::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
               const caffe::LayerParameter& weight) {
  auto relu = new tars::ReluT;
  if (parameters.relu_param().has_negative_slope()) {
    relu->slope = parameters.relu_param().negative_slope();
  } else {
    relu->slope = 0.0f;
  }
  dstOp->main.value = relu;
}

static OpConverterRegister<Relu> a("ReLU");

void Relu6::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                const caffe::LayerParameter& weight) {
  auto relu6 = new tars::Relu6T;
  dstOp->main.value = relu6;
}

static OpConverterRegister<Relu6> b("ReLU6");

class PRelu : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight) {
    auto relu = new tars::PReluT;
    auto v0w = &weight;
    DCHECK(v0w->blobs_size() >= 1) << "caffemodel error!";
    const caffe::BlobProto& slope_blob = v0w->blobs(0);
    relu->slopeCount = slope_blob.data_size();
    relu->slope.resize(relu->slopeCount);

    memcpy(relu->slope.data(), slope_blob.data().data(),
           sizeof(float) * relu->slopeCount);
    dstOp->main.value = relu;
  }
  PRelu() {}
  virtual ~PRelu() {}
  virtual tars::OpType opType() { return tars::OpType_PReLU; }
  virtual tars::OpParameter type() { return tars::OpParameter_PRelu; }
};

static OpConverterRegister<PRelu> __a("PReLU");

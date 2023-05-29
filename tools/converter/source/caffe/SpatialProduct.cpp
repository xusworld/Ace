//
//  SpatialProduct.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class SpatialProduct : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  SpatialProduct() {}
  virtual ~SpatialProduct() {}
  virtual tars::OpType opType() { return tars::OpType_SpatialProduct; }
  virtual tars::OpParameter type() { return tars::OpParameter_NONE; }
};

void SpatialProduct::run(tars::OpT* dstOp,
                         const caffe::LayerParameter& parameters,
                         const caffe::LayerParameter& weight) {}
static OpConverterRegister<SpatialProduct> a("SpatialProduct");

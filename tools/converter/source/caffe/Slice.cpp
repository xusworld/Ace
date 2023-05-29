//
//  Slice.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
class Slice : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Slice() {}
  virtual ~Slice() {}
  virtual tars::OpType opType() { return tars::OpType_Slice; }
  virtual tars::OpParameter type() { return tars::OpParameter_Slice; }
};

void Slice::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                const caffe::LayerParameter& weight) {
  auto slice = new tars::SliceT;
  dstOp->main.value = slice;
  auto c = parameters.slice_param();
  slice->axis = c.axis();
  for (int i = 0; i < c.slice_point_size(); ++i) {
    slice->slicePoints.push_back(c.slice_point(i));
  }
}
static OpConverterRegister<Slice> a("Slice");

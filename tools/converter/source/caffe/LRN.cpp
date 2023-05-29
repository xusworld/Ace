//
//  LRN.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Lrn : public OpConverter {
 public:
  void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
           const caffe::LayerParameter& weight);
  virtual tars::OpType opType() { return tars::OpType_LRN; }
  virtual tars::OpParameter type() { return tars::OpParameter_LRN; }
};

void Lrn::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
              const caffe::LayerParameter& weight) {
  tars::LRNT* lrn = new tars::LRNT;
  dstOp->main.value = lrn;

  auto caffeLrn = parameters.lrn_param();
  lrn->alpha = caffeLrn.alpha();
  lrn->beta = caffeLrn.beta();
  lrn->localSize = caffeLrn.local_size();
  lrn->regionType = caffeLrn.norm_region();
}

static OpConverterRegister<Lrn> a("LRN");
static OpConverterRegister<Lrn> _a("CuDNNLRNCrossChannel");

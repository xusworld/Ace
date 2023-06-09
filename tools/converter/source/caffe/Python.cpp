//
//  Python.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "OpConverter.hpp"
using namespace std;

class Python : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  virtual tars::OpType opType() { return tars::OpType_Proposal; }
  virtual tars::OpParameter type() { return tars::OpParameter_Proposal; }
};

void Python::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                 const caffe::LayerParameter& weight) {
  auto proposal = new tars::ProposalT;
  auto Par = parameters.python_param();
  if (!Par.has_param_str()) {
    proposal->featStride = 16;
  } else {
    const string& StrideStr = Par.param_str();
    const string numb = StrideStr.substr(StrideStr.find(':') + 1);
    proposal->featStride = (int)atof(numb.c_str());
  }
  proposal->baseSize = 8;
  proposal->preNmsTopN = 300;
  proposal->afterNmsTopN = 32;
  proposal->nmsThreshold = 0.7f;
  proposal->minSize = 3;
  proposal->ratios = std::unique_ptr<tars::BlobT>(new tars::BlobT);
  proposal->ratios->dataType = tars::DataType_DT_FLOAT;
  proposal->ratios->float32s.resize(3);
  proposal->ratios->float32s[0] = 0.5f;
  proposal->ratios->float32s[1] = 1.0f;
  proposal->ratios->float32s[2] = 2.0f;
  proposal->ratios->dims.push_back(3);

  proposal->scales = std::unique_ptr<tars::BlobT>(new tars::BlobT);
  proposal->scales->dataType = tars::DataType_DT_FLOAT;
  proposal->scales->float32s.resize(3);
  proposal->scales->float32s[0] = 8.0f;
  proposal->scales->float32s[1] = 16.0f;
  proposal->scales->float32s[2] = 32.0f;
  proposal->scales->dims.push_back(3);

  dstOp->main.value = proposal;
}

static OpConverterRegister<Python> a("Python");

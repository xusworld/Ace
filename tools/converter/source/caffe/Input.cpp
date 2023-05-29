//
//  Input.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include "logkit.h"

class Input : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Input() {}
  virtual ~Input() {}
  virtual tars::OpType opType() { return tars::OpType_Input; }
  virtual tars::OpParameter type() { return tars::OpParameter_Input; }
};

void Input::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                const caffe::LayerParameter& weight) {
  tars::InputT* input = new tars::InputT;
  std::vector<int> dims;
  auto inputParametar = parameters.input_param();
  DCHECK(inputParametar.shape_size() == 1);
  auto shape = inputParametar.shape(0);
  for (int i = 0; i < shape.dim_size(); ++i) {
    dims.push_back(shape.dim(i));
  }
  input->dims = dims;
  dstOp->main.value = input;
}

static OpConverterRegister<Input> a("Input");

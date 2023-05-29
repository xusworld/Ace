//
//  Crop.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include "logkit.h"

class Crop : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);

  Crop() {}
  virtual ~Crop() {}

  virtual tars::OpType opType() { return tars::OpType_Crop; }
  virtual tars::OpParameter type() { return tars::OpParameter_Crop; }
};

void Crop::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
               const caffe::LayerParameter& weight) {
  auto cropParam = new tars::CropT;

  auto& caffeCrop = parameters.crop_param();

  if (caffeCrop.has_axis()) {
    cropParam->axis = caffeCrop.axis();
  } else {
    cropParam->axis = 2;
  }
  const int offsetSize = caffeCrop.offset_size();
  DCHECK(offsetSize >= 1) << "crop offset error";
  cropParam->offset.resize(offsetSize);
  for (int i = 0; i < offsetSize; ++i) {
    cropParam->offset[i] = caffeCrop.offset().data()[i];
  }

  dstOp->main.value = cropParam;
}

static OpConverterRegister<Crop> c("Crop");

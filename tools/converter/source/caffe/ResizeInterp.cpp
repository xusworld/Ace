//
//  ResizeInterp.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Upsample : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Upsample() {}
  virtual ~Upsample() {}
  virtual tars::OpType opType() { return tars::OpType_Resize; }
  virtual tars::OpParameter type() { return tars::OpParameter_Resize; }
};

void Upsample::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight) {
  auto resize = new tars::ResizeT;
  dstOp->main.value = resize;
  auto& r = parameters.upsample_param();
  resize->xScale = r.scale();
  resize->yScale = r.scale();
}
static OpConverterRegister<Upsample> ___a("Upsample");

class Resize : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Resize() {}
  virtual ~Resize() {}
  virtual tars::OpType opType() { return tars::OpType_Resize; }
  virtual tars::OpParameter type() { return tars::OpParameter_Resize; }
};

void Resize::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                 const caffe::LayerParameter& weight) {
  auto resize = new tars::ResizeT;
  dstOp->main.value = resize;
  auto& r = parameters.img_size_param();
  resize->xScale = r.x_scaling();
  resize->yScale = r.y_scaling();
}
static OpConverterRegister<Resize> a("Resize");

class Interp : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);
  Interp() {}
  virtual ~Interp() {}
  virtual tars::OpType opType() { return tars::OpType_Interp; }
  virtual tars::OpParameter type() { return tars::OpParameter_Interp; }
};

void Interp::run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                 const caffe::LayerParameter& weight) {
  auto resize = new tars::InterpT;
  dstOp->main.value = resize;
  auto& Par = parameters.interp_param();
  resize->widthScale = 1.0f;
  resize->heightScale = 1.0f;
  if (Par.has_shrink_factor()) {
    resize->widthScale = (float)(1.0 / Par.shrink_factor());
    resize->heightScale = (float)(1.0 / Par.shrink_factor());
  }
  if (Par.has_zoom_factor()) {
    resize->widthScale = (float)(resize->widthScale + 1.0 / Par.zoom_factor());
    resize->heightScale =
        (float)(resize->heightScale + 1.0 / Par.zoom_factor());
  }
  if (Par.has_height()) resize->outputHeight = Par.height();
  if (Par.has_width()) resize->outputWidth = Par.width();
  resize->resizeType = 2;
  resize->alignCorners = true;
}

static OpConverterRegister<Interp> b("Interp");

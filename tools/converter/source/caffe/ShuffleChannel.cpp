//
//  ShuffleChannel.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include "logkit.h"

class ShuffleChannel : public OpConverter {
 public:
  virtual void run(tars::OpT* dstOp, const caffe::LayerParameter& parameters,
                   const caffe::LayerParameter& weight);

  virtual tars::OpType opType() { return tars::OpType_Plugin; }
  virtual tars::OpParameter type() { return tars::OpParameter_Plugin; }
};

void ShuffleChannel::run(tars::OpT* dstOp,
                         const caffe::LayerParameter& parameters,
                         const caffe::LayerParameter& weight) {
  auto plugin = new tars::PluginT;
  plugin->type = "ShuffleChannel";
  plugin->attr.resize(1);
  plugin->attr[0].reset(new tars::AttributeT);
  plugin->attr[0]->tensor.reset(new tars::BlobT);
  auto blob = plugin->attr[0]->tensor.get();
  blob->int32s = {1};
  if (parameters.has_shuffle_channel_param()) {
    blob->int32s = {(int)parameters.shuffle_channel_param().group()};
  }
  dstOp->main.value = plugin;
}

static OpConverterRegister<ShuffleChannel> a("ShuffleChannel");

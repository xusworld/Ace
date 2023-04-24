//
//  PluginFactory.cpp
//  MNN
//
//  Created by MNN on b'2020/08/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "BinaryPlugin.hpp"
#include "CommonPlugin.hpp"
#include "PluginFactory.hpp"
#include "PreluPlugin.hpp"
#include "ace_generated.h"

using namespace ace;

void* MNNTRTCreatePlugion(const void* opRaw, const void* extraInfo) {
  auto op = (ace::Op*)opRaw;
  auto plugin = (const MNNTRTPlugin::PluginT*)extraInfo;
  return new CommonPlugin(op, plugin);
}

void* MNNTRTCreatePlugionSerial(const char* layerName, const void* serialData,
                                size_t serialLength) {
  return new CommonPlugin(serialData, serialLength);
}

//
//  CastPlugin.cpp
//  MNN
//
//  Created by MNN on b'2020/08/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <ace/MNNDefine.h>

#include "CastPlugin.hpp"
namespace ace {

CastPlugin::CastPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin) {
  auto Info = plugin->main_as_OneHotInfo();
  mCount = Info->outerSize();
}

CastPlugin::~CastPlugin() {}

int CastPlugin::onEnqueue(int batchSize, const void* const* inputs,
                          void** outputs, void*, nvinfer1::DataType dataType,
                          cudaStream_t stream) {
  int size = 0;
  if (dataType == nvinfer1::DataType::kFLOAT) {
    size = mCount * sizeof(float);
  } else {
    size = mCount * sizeof(__half);
  }

  auto status =
      cudaMemcpy(outputs[0], inputs[0], size, cudaMemcpyDeviceToDevice);
  MNN_ASSERT(0 == status);
}

};  // namespace ace

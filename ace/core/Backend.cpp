//
//  Backend.cpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include <mutex>

#include "ace/types.h"
#include "ace_generated.h"
#include "backend/cpu/CPUTensorConvert.hpp"
#include "core/Backend.hpp"
#include "core/Macro.h"

namespace ace {

void registerBackend();

static std::map<DeviceType, std::pair<const RuntimeCreator*, bool>>&
GetExtraCreator() {
  static std::once_flag gInitFlag;
  static std::map<DeviceType, std::pair<const RuntimeCreator*, bool>>*
      gExtraCreator;
  std::call_once(gInitFlag, [&]() {
    gExtraCreator =
        new std::map<DeviceType, std::pair<const RuntimeCreator*, bool>>;
  });
  return *gExtraCreator;
}

const RuntimeCreator* MNNGetExtraRuntimeCreator(DeviceType type) {
  registerBackend();

  auto& gExtraCreator = GetExtraCreator();
  auto iter = gExtraCreator.find(type);
  if (iter == gExtraCreator.end()) {
    return nullptr;
  }
  if (!iter->second.second) {
    return iter->second.first;
  }
  Backend::Info info;
  info.type = type;
  std::shared_ptr<Runtime> bn(iter->second.first->onCreate(info));
  if (nullptr != bn.get()) {
    return iter->second.first;
  }
  return nullptr;
}

bool MNNInsertExtraRuntimeCreator(DeviceType type,
                                  const RuntimeCreator* creator,
                                  bool needCheck) {
  auto& gExtraCreator = GetExtraCreator();
  if (gExtraCreator.find(type) != gExtraCreator.end()) {
    MNN_ASSERT(false && "duplicate type");
    return false;
  }
  gExtraCreator.insert(
      std::make_pair(type, std::make_pair(creator, needCheck)));
  return true;
}

bool MNNCPUCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) {
  auto& srcBuffer = srcTensor->buffer();
  auto& dstBuffer = dstTensor->buffer();

  MNN_ASSERT(srcBuffer.dimensions == dstBuffer.dimensions);
  MNN_ASSERT(srcBuffer.type == dstBuffer.type);
  if (nullptr == srcBuffer.host || nullptr == dstBuffer.host) {
    return false;
  }
  auto code = CPUTensorConverter::convert(srcTensor, dstTensor);
  if (NO_ERROR != code) {
    MNN_ERROR("Error in CPUBackend::onCopyBuffer\n");
  }
  return true;
}

}  // namespace ace

//
//  BackendRegister.cpp
//  MNN
//
//  Created by MNN on 2019/05/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <mutex>

#include "Macro.h"
#include "geometry/GeometryComputer.hpp"
#include "shape/SizeComputer.hpp"
#ifdef MNN_INTERNAL_ENABLED
#include "internal/logging/Log.hpp"
#endif
namespace tars {
extern void registerCPURuntimeCreator();

static std::once_flag s_flag;
void registerBackend() {
  std::call_once(s_flag, [&]() {
#ifdef MNN_INTERNAL_ENABLED
    LogInit();
#endif
    registerCPURuntimeCreator();
#ifndef MNN_BUILD_MINI
    SizeComputerSuite::init();
    GeometryComputer::init();
#endif
  });
}
}  // namespace tars

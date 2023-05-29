//
//  AVX2Functions.hpp
//  MNN
//
//  Created by MNN on b'2021/05/17'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef AVX2Functions_hpp
#define AVX2Functions_hpp
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "core/Macro.h"
#include "cpu_id.h"
#include "device/cpu/compute/CommonOptFunction.h"
#include "device/cpu/compute/Int8FunctionsOpt.h"

namespace tars {
class AVX2Functions {
 public:
  static bool init(int flags);
  static CoreFunctions* get();
  static CoreInt8Functions* getInt8();
};
};  // namespace tars

#endif

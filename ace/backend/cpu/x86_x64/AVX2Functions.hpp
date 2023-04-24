#ifndef AVX2Functions_hpp
#define AVX2Functions_hpp
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "cpu_id.h"

namespace ace {
class AVX2Functions {
 public:
  static bool init(int flags);
  static CoreFunctions* get();
};
};  // namespace ace

#endif

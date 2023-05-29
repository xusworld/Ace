#ifndef BF16Functions_hpp
#define BF16Functions_hpp
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../compute/CommonOptFunction.h"
#include "core/Macro.h"
namespace tars {
class BF16Functions {
 public:
  static bool init();
  static CoreFunctions* get();
};
};  // namespace tars

#endif

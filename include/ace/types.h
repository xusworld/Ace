#pragma once

#include <stddef.h>
#include <stdint.h>

namespace ace {

enum class DeviceType : int {
  INVALID = -1,
  X86 = 0,
  X86_EXTENSION = 1,
  CUDA = 2,
  ALL = 3
};

enum class Precision : int {
  INVALID = -1,
  INT4 = 0,
  INT8 = 1,
  FP16 = 2,
  FP32 = 3,
  FP64 = 4
};

enum class OpRunType : int {
  INVALID = -1,
  SYNC = 1,  ///< the net exec synchronous (for GPU, means single-stream)
  ASYNC = 2  ///< ASYNC the net exec asynchronous (for GPU, means mutli-stream)
};

struct BackendConfig {
  enum MemoryMode { Memory_Normal = 0, Memory_High, Memory_Low };

  MemoryMode memory = Memory_Normal;

  enum PowerMode { Power_Normal = 0, Power_High, Power_Low };

  PowerMode power = Power_Normal;

  enum PrecisionMode { Precision_Normal = 0, Precision_High, Precision_Low };

  PrecisionMode precision = Precision_Normal;

  /** user defined context */
  union {
    void* sharedContext = nullptr;
    size_t flags;  // Valid for CPU Backend
  };
};

}  // namespace ace
#pragma once

#include "framework/device/memory_pool.h"
#include "framework/device/thread_pool.h"
#include "ir/types_generated.h"
#include "runtime.h"

namespace ace {
namespace device {

class CpuRuntime : public Runtime {
 public:
  using event_t = void*;
  using stream_t = void*;

 public:
  CpuRuntime() : type_(RuntimeType::CPU) {}
  ~CpuRuntime() = default;

 protected:
  RuntimeType type_ = RuntimeType::CPU;
  MemoryPool memory_pool_;
  ThreadPool thread_pool_;
};

}  // namespace device
}  // namespace ace
#pragma once

#include "core/runtime.h"

namespace ace {
namespace device {

class CudaDevice : public Runtime {
 public:
  using event_t = void*;
  using stream_t = void*;

 public:
  CudaDevice() : type_(RuntimeType::CUDA) {}
  ~CudaDevice() = default;

 protected:
  RuntimeType type_;
};

}  // namespace device
}  // namespace ace
#pragma once

#include <cstdint>

#include "allocator.h"

namespace ace {
namespace device {

class CpuAllocator : public Allocator {
 public:
  CpuAllocator() = default;
  virtual ~CpuAllocator() = default;

  virtual Status allocate(const DataType dtype, const int32_t size,
                          void **ptr) override;

  virtual Status allocate(const int32_t bytes, void **ptr) override;

  virtual Status release(void *ptr) override;

  virtual Status reset(const DataType dtype, const size_t val,
                       const int32_t bytes, void *ptr) override;

  static CpuAllocator *Get();

 private:
  static CpuAllocator *alloc_;
};

}  // namespace device
}  // namespace ace
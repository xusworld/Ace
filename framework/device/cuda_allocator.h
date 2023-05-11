#pragma once

#include "allocator.h"

namespace ace {
namespace device {

class CudaAllocator : public Allocator {
 public:
  CudaAllocator() = default;
  virtual ~CudaAllocator() = default;

  virtual Status allocate(const DataType dtype, const int32_t size,
                          void **ptr) override;

  virtual Status allocate(const int32_t bytes, void **ptr) override;

  virtual Status release(void *ptr) override;

  static CudaAllocator *Get();

 private:
  static CudaAllocator *alloc_;
};

}  // namespace device
}  // namespace ace
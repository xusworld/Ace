#include <cuda_runtime.h>
#include <glog/logging.h>
#include <mm_malloc.h>

#include <cstdint>
#include <cstdlib>

#include "cuda_allocator.h"
#include "utils.h"
namespace ace {
namespace device {

Status CudaAllocator::allocate(const DataType dtype, const int32_t size,
                               void **ptr) {
  const auto bytes = GetBufferBytes(dtype, size);
  CUDA_CHECK(cudaMalloc(&ptr, bytes));
  *ptr = reinterpret_cast<char *>(ptr);

  CHECK(*ptr != NULL) << "malloc return a null pointer, please check";
  return Status::OK();
}

Status CudaAllocator::allocate(const int32_t bytes, void **ptr) {
  CUDA_CHECK(cudaMalloc(&ptr, bytes));
  *ptr = reinterpret_cast<char *>(ptr);

  CHECK(*ptr != NULL) << "malloc return a null pointer, please check";
  return Status::OK();
}

Status CudaAllocator::release(void *ptr) {
  if (ptr != nullptr) {
    CUDA_CHECK(cudaFree(ptr));
  } else {
    LOG(INFO) << "ptr is a null pointer, please check.";
  }

  return Status::OK();
}

CudaAllocator *CudaAllocator::Get() {
  if (alloc_ == nullptr) {
    alloc_ = new CudaAllocator;
  }

  return alloc_;
}

CudaAllocator *CudaAllocator::alloc_ = nullptr;

}  // namespace device
}  // namespace ace
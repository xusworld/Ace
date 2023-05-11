#pragma once

#include <map>
#include <unordered_map>

#include "../framework/core/status.h"
#include "allocator.h"
#include "framework/device/cpu_allocator.h"
#include "framework/device/cuda_allocator.h"
#include "types.h"

namespace ace {
namespace device {

class MemoryManager {
 public:
  using Stream = void*;

  // acquire a buffer from memory pool
  static Status acuqire(void** ptr, const int32_t size) {
    return Status::UNIMPLEMENTED("acquire interface is not implemented yet.");
  }

  // release a buffer from memroy pool
  static Status release(void* ptr) {
    return Status::UNIMPLEMENTED("release interface is not implemented yet.");
  }

  // reset a buffer in memory pool
  static Status reset(void* ptr, const int32_t val, const int32_t size) {
    return Status::UNIMPLEMENTED("reset interface is not implemented yet.");
  }

  // memcpy
  static Status sync_memcpy(void* dst, size_t dst_offset, int dst_id,
                            const void* src, size_t src_offset, int src_id,
                            size_t count, MemcpyKind) {
    return Status::UNIMPLEMENTED(
        "sync_memcpy interface is not implemented yet.");
  }

  // asynchronize memcpy
  static Status async_memcpy(void* dst, size_t dst_offset, int dst_id,
                             const void* src, size_t src_offset, int src_id,
                             size_t count, MemcpyKind) {
    return Status::UNIMPLEMENTED(
        "async_memcpy interface is not implemented yet.");
  }

  //  memcpy peer to peer, for device memory copy between different devices
  static Status sync_memcpy_p2p(void* dst, size_t dst_offset, int dst_id,
                                const void* src, size_t src_offset, int src_id,
                                size_t count) {
    return Status::UNIMPLEMENTED(
        "sync_memcpy_p2p interface is not implemented yet.");
  }

  // asynchronize memcpy peer to peer, for device memory copy between different
  // devices
  static Status async_memcpy_p2p(void* dst, size_t dst_offset, int dst_id,
                                 const void* src, size_t src_offset, int src_id,
                                 size_t count, Stream stream) {
    return Status::UNIMPLEMENTED(
        "async_memcpy_p2p interface is not implemented yet.");
  }
};

namespace {

using BlockList = std::multimap<int, void*>;

}

class MemoryPool {
 public:
  explicit MemoryPool(RuntimeType rtype = RuntimeType::CPU) : rtype_(rtype) {
    if (rtype_ == RuntimeType::CUDA) {
      allocator_ = CudaAllocator::Get();
    }
  }

  MemoryPool(Allocator* allocator, RuntimeType rtype = RuntimeType::CPU)
      : allocator_(allocator), rtype_(rtype) {}

  ~MemoryPool() = default;

  // acquire a buffer
  Status acuqire(void** ptr, const int32_t bytes);

  // release a buffer
  Status release(void* ptr);

  // release all buffer
  Status release_all(bool is_free_memory);

  int32_t free_blocks_num() const { return free_blocks_.size(); }

  int32_t used_blocks_num() const { return used_blocks_.size(); }

  // TODO(xusworld): 优化内存池碎片

 private:
  Status clear();

 private:
  BlockList used_blocks_;
  BlockList free_blocks_;
  RuntimeType rtype_ = RuntimeType::CPU;
  Allocator* allocator_ = CpuAllocator::Get();
};

}  // namespace device
}  // namespace ace
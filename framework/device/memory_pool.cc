#include "framework/core/status.h"
#include "glog/logging.h"
#include "memory_pool.h"

namespace ace {
namespace device {

// acquire a buffer
Status MemoryPool::acuqire(void** ptr, const int32_t bytes) {
  auto iter = free_blocks_.lower_bound(bytes);
  if (iter == free_blocks_.end()) {
    CHECK_EQ(allocator_->allocate(bytes, ptr), Status::OK())
        << "Malloc new memory failed, please check.";
    // mark as a used memory block
    used_blocks_.emplace(bytes, ptr);
  } else {
    *ptr = iter->second;
    used_blocks_.emplace(iter->first, iter->second);
    free_blocks_.erase(iter);
  }

  return Status::OK();
}

// release a buffer
Status MemoryPool::release(void* ptr) {
  for (auto& block : used_blocks_) {
    if (block.second == ptr) {
      free_blocks_.emplace(block);
      used_blocks_.erase(block.first);
      return Status::OK();
    }
  }
  return Status::ERROR("can't find pointer in memory pool, please check.");
}

// release all buffer
Status MemoryPool::release_all(bool is_free_memory) {
  if (is_free_memory) {
    return this->clear();
  }

  for (auto& block : used_blocks_) {
    free_blocks_.emplace(block);
  }
  used_blocks_.clear();

  return Status::UNIMPLEMENTED("release interface is not implemented yet.");
}

Status MemoryPool::clear() {
  for (auto& block : used_blocks_) {
    allocator_->release(block.second);
  }
  used_blocks_.clear();

  for (auto& block : free_blocks_) {
    allocator_->release(block.second);
  }
  free_blocks_.clear();

  return Status::OK();
}

}  // namespace device
}  // namespace ace
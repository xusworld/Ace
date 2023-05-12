#include <glog/logging.h>

#include "buffer.h"
#include "status.h"

namespace ace {

Status Buffer::allocate(const size_t size) {
  if (buff_ != nullptr) {
    LOG(WARNING) << "Not a empty buffer, please check";
  }

  if (size_ <= 0) {
    LOG(ERROR) << "Try to allocate " << size_
               << " bytes memory space, which is < 0";
  }

  allocator_->allocate(dtype_, size_, &buff_);
  if (buff_ == nullptr) {
    LOG(ERROR) << "Allocate memory failed.";
    return Status::FATAL("Allocate memory failed.");
  }

  device_id_ = this->allocator_->device_id();
  capacity_ = size_;
  return Status::OK();
}

Status Buffer::allocate(const std::vector<int32_t> &dims) {
  const int32_t size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  return this->allocate(size);
}

Status Buffer::allocate(const TensorShape &shape) {
  const int32_t size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  return this->allocate(size);
}

Status Buffer::realloc(const size_t size) {
  if (size > capacity_) {
    if (reuseable_) {
      LOG(ERROR) << "realloc error.";
    } else {
      this->allocator_->allocate(dtype_, size, &buff_);
      capacity_ = size;
    }
  }
  size_ = size;
  return Status::OK();
}

Status Buffer::realloc(const std::vector<int32_t> &dims) {
  const int32_t size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  return this->realloc(size);
}

Status Buffer::realloc(const TensorShape &shape) {
  const int32_t size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  return this->realloc(size);
}

Status Buffer::clear() {
  if (!reuseable_) {
    size_ = 0;
    capacity_ = 0;
    allocator_->release(buff_);
  }
  buff_ = nullptr;
  return Status::OK();
}

}  // namespace ace
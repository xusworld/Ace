#pragma once

#include <cstddef>

#include "../framework/core/status.h"
#include "allocator.h"
#include "cpu_allocator.h"
#include "cuda_allocator.h"
#include "ir/types_generated.h"
#include "runtime.h"
#include "tensor_shape.h"
#include "types.h"

namespace ace {
namespace device {

// TODO: CUDA cross devices memory copy

class Buffer {
 public:
  Buffer(const int32_t size) : size_(size) { this->allocate(size); }

  Buffer(RuntimeType rtype, const int32_t size) : rtype_(rtype) {
    if (rtype == RuntimeType::CUDA) {
      allocator_ = CudaAllocator::Get();
    }

    this->allocate(size);
  }

  Buffer(RuntimeType rtype, DataType dtype, const int32_t size)
      : rtype_(rtype), dtype_(dtype), size_(size) {
    if (rtype == RuntimeType::CUDA) {
      allocator_ = CudaAllocator::Get();
    }

    this->allocate(size);
  }

  Buffer(const Buffer& buffer) {
    CHECK_GT(buffer.size_, 0) << " Buffer Constructer got a empty buffer.";
    size_ = buffer.size_;
    // buffers on the same device
    if (this->allocator_->device_id() == buffer.device_id_) {
      buff_ = buffer.buff_;
      reuseable_ = buffer.reuseable_;
      capacity_ = buffer.capacity_;
    } else {
      reuseable_ = false;
      this->realloc(buffer.size_);
      // copy to other device
    }
  }

  ~Buffer();

  Status allocate(const size_t size);
  Status allocate(const std::vector<int32_t>& dims);
  Status allocate(const TensorShape& shape);

  Status realloc(const size_t size);
  Status realloc(const std::vector<int32_t>& dims);
  Status realloc(const TensorShape& shape);

  Status clear();

  // buffer pointer
  template <typename T>
  const T* data() const {
    return reinterpret_cast<const T*>(buff_);
  }

  template <typename T>
  T* mutable_data() {
    return reinterpret_cast<T*>(buff_);
  }

  // real buffer size
  int32_t size() const { return size_; }

  // buffer capacity
  int32_t capacity() const { return capacity_; };

  void SetReUseable(const bool flag) { reuseable_ = flag; }

 private:
  Allocator* allocator_ = CpuAllocator::Get();
  RuntimeType rtype_ = RuntimeType::CPU;
  DataType dtype_ = DataType_FLOAT_32;

  int32_t device_id_;
  void* buff_;
  size_t size_ = 0;
  size_t capacity_ = 0;
  bool reuseable_ = false;
};

// /**
//  * \brief assigned function, ptop memcpy is called if src is in different
//  * device
//  */
// Buffer& operator=(Buffer<DeviceType>& buf) {
//   this->_count = buf._count;
//   this->_id = API::get_device_id();
//   if (buf._id == this->_id) {
//     this->_data = buf._data;
//     this->_capacity = this->_count;
//     this->_own_data = false;
//   } else {
//     this->_own_data = true;
//     SABER_CHECK(this->re_alloc(buf._count));
//     API::sync_memcpy_p2p(this->_data, 0, this->_id, buf.get_data(), 0,
//     buf._id,
//                          buf._count);
//   }
//   return *this;
// }

// int shared_from(Buffer<DeviceType>& buf) {
//   _count = buf._count;
//   _id = API::get_device_id();
//   if (buf._id == _id) {
//     _data = buf._data;
//     _capacity = _count;
//     _own_data = false;
//     return 1;
//   } else {
//     _own_data = true;
//     SABER_CHECK(re_alloc(buf._count));
//     API::sync_memcpy_p2p(_data, 0, _id, buf.get_data(), 0, buf._id,
//     buf._count); return 0;
//   }
// }

// ~Buffer() { clean(); }

// Status mem_set(int c, size_t size) {
//   if (!_own_data || _count < size) {
//     return SaberOutOfAuthority;
//   }
//   API::mem_set(_data, c, size);
//   return SaberSuccess;
// }

// template <typename DeviceType_t>
// Status sync_copy_from(Buffer<DeviceType_t>& buf) {
//   CHECK_GE(_capacity, buf.get_count());
//   // 类型萃取
//   typedef TargetWrapper<DeviceType_t> API_t;
//   typedef
//       typename DeviceTypeTraits<DeviceType>::target_category target_category;
//   typedef typename DeviceTypeTraits<DeviceType>::target_type
//   target_type_this; typedef typename
//   DeviceTypeTraits<DeviceType_t>::target_type target_type_t; typedef typename
//   IF<std::is_same<target_type_this, target_type_t>::value,
//                       __HtoH, __DtoH>::Type then_type;
//   typedef typename IF<std::is_same<target_type_this, target_type_t>::value,
//                       __DtoD, __HtoD>::Type else_type;
//   typedef typename IF<std::is_same<target_category, __host_target>::value,
//                       then_type, else_type>::Type flag_type;

//   typedef typename IF<std::is_same<target_category, __host_target>::value,
//                       API_t, API>::Type process_API;

//   process_API::sync_memcpy(_data, 0, _id, buf.get_data(), 0, buf.get_id(),
//                            buf.get_count(), flag_type());

//   return SaberSuccess;
// }

// template <typename dtype>
// Status from_vector(const std::vector<dtype>& data) {
//   typedef
//       typename DeviceTypeTraits<DeviceType>::target_category target_category;
//   typedef typename DeviceTypeTraits<DeviceType>::target_type
//   target_type_this; typedef __HtoH then_type; typedef __HtoD else_type;
//   typedef typename IF<std::is_same<target_category, __host_target>::value,
//                       then_type, else_type>::Type flag_type;

//   size_t vec_cap = data.size() * sizeof(dtype);
//   if (_capacity < vec_cap) {
//     alloc(vec_cap);
//   }
//   API::sync_memcpy(_data, 0, _id, data.data(), 0, 0, vec_cap, flag_type());

//   return SaberSuccess;
// }

// template <typename DeviceType_dst, typename DeviceType_src>
// static inline int MemShare(std::shared_ptr<Buffer<DeviceType_dst>>& dst,
//                            const std::shared_ptr<Buffer<DeviceType_src>>&
//                            src,
//                            __DtoD) {
//   // LOG(INFO) << "shared D2D";
//   if (dst->get_id() == src->get_id()) {
//     dst = src;
//     return 1;
//   }
//   // LOG(INFO) << "copied D2D";
//   SABER_CHECK(dst->re_alloc(src->get_count()));
//   SABER_CHECK(dst->sync_copy_from(*src));
//   return 0;
// }

// template <typename DeviceType_dst, typename DeviceType_src>
// static inline int MemShare(std::shared_ptr<Buffer<DeviceType_dst>>& dst,
//                            const std::shared_ptr<Buffer<DeviceType_src>>&
//                            src,
//                            __HtoD) {
//   // LOG(INFO) << "copied H2D";
//   SABER_CHECK(dst->re_alloc(src->get_count()));
//   SABER_CHECK(dst->sync_copy_from(*src));
//   return 0;
// }

// template <typename DeviceType_dst, typename DeviceType_src>
// static inline int MemShare(std::shared_ptr<Buffer<DeviceType_dst>>& dst,
//                            const std::shared_ptr<Buffer<DeviceType_src>>&
//                            src,
//                            __HtoH) {
//   // LOG(INFO) << "shared H2H";
//   dst = src;
//   return 1;
// }

// template <typename DeviceType_dst, typename DeviceType_src>
// static inline int MemShare(std::shared_ptr<Buffer<DeviceType_dst>>& dst,
//                            const std::shared_ptr<Buffer<DeviceType_src>>&
//                            src,
//                            __DtoH) {
//   // LOG(INFO) << "copied D2H";
//   SABER_CHECK(dst->re_alloc(src->get_count()));
//   SABER_CHECK(dst->sync_copy_from(*src));
//   return 0;
// }

// template <typename DeviceType_dst, typename DeviceType_src>
// static inline int BufferMemShare(
//     std::shared_ptr<Buffer<DeviceType_dst>>& dst,
//     const std::shared_ptr<Buffer<DeviceType_src>>& src) {
//   typedef
//       typename DeviceTypeTraits<DeviceType_dst>::target_type target_type_dst;
//   typedef
//       typename DeviceTypeTraits<DeviceType_src>::target_type target_type_src;
//   typedef typename DeviceTypeTraits<DeviceType_dst>::target_category
//       target_category_dst;

//   typedef typename IF<std::is_same<target_type_dst, target_type_src>::value,
//                       __HtoH, __DtoH>::Type then_type;
//   typedef typename IF<std::is_same<target_type_dst, target_type_src>::value,
//                       __DtoD, __HtoD>::Type else_type;
//   typedef typename IF<std::is_same<target_category_dst,
//   __host_target>::value,
//                       then_type, else_type>::Type flag_type;
//   CHECK_EQ(src == nullptr, false) << "input buffer is null!";
//   if (!dst) {
//     dst = std::make_shared<Buffer<DeviceType_dst>>();
//   }
//   return MemShare(dst, src, flag_type());
// }
// * /

}  // namespace device

}  // namespace ace

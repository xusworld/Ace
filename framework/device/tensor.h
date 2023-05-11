#pragma once

#include <memory>

#include "buffer.h"
#include "events.h"
#include "ir/types_generated.h"
#include "tensor_shape.h"
#include "type_traits_extend.h"
#include "types.h"
#include "utils.h"

namespace ace {
namespace device {

// template <typename TargetType>
// class Tensor {
//  public:
//   typedef typename DataTraitBase<TargetType>::PtrDtype BaseDtype;

//   typedef
//       typename TargetTypeTraits<TargetType>::target_category target_category;
//   typedef typename TargetTypeTraits<TargetType>::target_type target_type;
//   typedef TargetWrapper<TargetType> API;

//   /**
//    * \brief Constructor with allocated data ptr and entire memory shape.
//    */
//   //! now only support fp32 data pointer
//   template <typename TargetType_t>
//   Tensor(typename DataTraitBase<TargetType_t>::PtrDtype data_ptr,
//          TargetType_t target, int id, Shape shape, DataType type = AK_FLOAT)
//          {
//     _shape = shape;
//     _valid_shape = shape;
//     _offset = Shape::zero(shape);
//     _dtype = type;
//     _type_len = type_length(type);
//     std::shared_ptr<Buffer<TargetType_t>> buf_from_date =
//         std::make_shared<Buffer<TargetType_t>>(data_ptr,
//                                                shape.count() * _type_len,
//                                                id);
//     BufferMemShare(_buf, buf_from_date);
//     _is_shared = true;
//     _is_subbuf = false;
//   }

class Tensor {
 public:
  Tensor(DataType dtype = DataType_FLOAT_32) : dtype_(dtype) {
    buff_ = std::make_shared<Buffer>();
  }

  Tensor(const TensorShape &shape, DataType dtype = DataType_FLOAT_32)
      : shape_(shape), valid_shape_(shape), dtype_(dtype) {
    offset_ = TensorShape::zero(shape);
    buff_ = std::make_shared<Buffer>(rtype_, dtype_, shape.size());
    is_shared_ = false;
  }

  Tensor(const Tensor &tensor) {
    shape_ = tensor.shape_;
    valid_shape_ = tensor.valid_shape_;
    offset_ = tensor.offset_;
    dtype_ = tensor.dtype_;
    dformat_ = tensor.dformat_;
    is_shared_ = tensor.is_shared_;
    buff_ = tensor.buff_;
  }

  ~Tensor() = default;

  TensorShape mutable_shape() const { return shape_; }
  const TensorShape shape() const { return shape_; };

  TensorShape mutable_valid_shape() const { return valid_shape_; }
  const TensorShape valid_shape() const { return valid_shape_; };

  TensorShape mutable_offset() const { return offset_; }
  const TensorShape offset() const { return offset_; };

  DataType dtype() const { return dtype_; }
  DataFormat dformat() const { return dformat_; }
  RuntimeType rtype() const { return rtype_; }

  int32_t size() const { return shape_.size(); }
  int32_t bytes() const { return shape_.size() * DataType2Bytes(dtype_); }
  int32_t capacity() const { return buff_->capacity(); }

  // TODO
  Status set_dtype() {}

  Status set_shape() {}

  Status realloc() {}

  Status reshape() {}

 private:
  // basic attributes
  DataType dtype_ = DataType_FLOAT_32;
  DataFormat dformat_ = DataFormat_NCHW;
  RuntimeType rtype_ = RuntimeType::CPU;

  // Represent the raw mem shape.
  TensorShape shape_;
  // Represent the mem you have right to access shape.
  TensorShape valid_shape_;
  // Represent the offset idx between _shape and _real_shape.
  TensorShape offset_;
  // buffer
  std::shared_ptr<Buffer> buff_ = nullptr;

  bool is_shared_ = false;
};

//   SaberStatus set_dtype(DataType type) {
//     _dtype = type;
//     _type_len = type_length(type);
//     if (_buf->get_capacity() < _shape.count() * _type_len) {
//       if (_is_shared || _is_subbuf) {
//         LOG(FATAL) << "tensor is shared, memory can not be re-alloced";
//         return SaberOutOfAuthority;
//       }
//       _buf_dtype = type;
//       _buf->re_alloc(_shape.count() * _type_len);
//     }
//     return SaberSuccess;
//   }

//   size_t get_dtype_size() const { return get_type_size(_dtype); }

//   DataType get_buf_dtype() const { return _buf_dtype; }

//   size_t get_buf_dtype_size() const { return get_type_size(_buf_dtype); }
//   /**
//    * \brief change tensor's layout and type
//    * @param layout
//    * @param data
//    * @return
//    */
//   SaberStatus set_layout(LayoutType layout, std::vector<int> data = {}) {
//     _valid_shape.set_layout(layout, data);
//     return SaberSuccess;
//   }
//   //    SaberStatus set_layout_without_shape(LayoutType layout) {
//   //        _valid_shape.set_layout_without_shape(layout);
//   //        return SaberSuccess;
//   //    }

//   LayoutType get_layout() const { return _valid_shape.get_layout(); }

//   /**
//    *  \brief only change the shape and valid shape, do nothing to memory
//    *  \param shape
//    *  \param valid_shape
//    *  \param offset
//    */
//   SaberStatus set_shape(Shape valid_shape, Shape shape = Shape(),
//                         Shape offset = Shape()) {
//     if (shape.dims() > 0) {
//       _shape = shape;
//     }
//     if (offset.dims() > 0 && _is_subbuf) {
//       _offset = offset;
//     }
//     CHECK_EQ(valid_shape > Shape::zero(valid_shape), true)
//         << "valid_shape size should > 0";
//     _valid_shape = valid_shape;

//     if (!_is_subbuf) {
//       if (_shape.count() <= _valid_shape.count()) {
//         _shape = _valid_shape;
//       }
//       _offset = Shape::zero(valid_shape);
//     } else {
//       if (_shape == Shape::zero(_valid_shape)) {
//         _shape = valid_shape;
//       }
//       //if (!(_valid_shape + _offset <= _shape)) { \
//                 return SaberInvalidValue; \
//             }
//       CHECK_EQ(_valid_shape + _offset <= _shape, true)
//           << "valid_shape + offet should <= shape";
//     }
//     return SaberSuccess;
//   }

//   /**
//    *  \brief only change the shape and valid shape, do nothing to memory
//    *  \param shape
//    *  \param valid_shape
//    *  \param offset
//    */
//   SaberStatus set_shape_without_layout(Shape valid_shape, Shape shape =
//   Shape(),
//                                        Shape offset = Shape()) {
//     if (shape.dims() > 0) {
//       _shape.set_shape_without_layout(shape);
//     }
//     if (offset.dims() > 0 && _is_subbuf) {
//       _offset.set_shape_without_layout(offset);
//     }
//     CHECK_EQ(valid_shape > Shape::zero(valid_shape), true)
//         << "valid_shape size should > 0";
//     _valid_shape.set_shape_without_layout(valid_shape);

//     if (!_is_subbuf) {
//       if (_shape.count() <= _valid_shape.count()) {
//         _shape = _valid_shape;
//       }
//       _offset = Shape::zero(valid_shape);
//     } else {
//       if (_shape == Shape::zero(_valid_shape)) {
//         _shape = valid_shape;
//       }
//       //if (!(_valid_shape + _offset <= _shape)) { \
//                 return SaberInvalidValue; \
//             }
//       CHECK_EQ(_valid_shape + _offset <= _shape, true)
//           << "valid_shape + offet should <= shape";
//     }
//     return SaberSuccess;
//   }

//   /**
//    *  \brief Free old buffer and alloc a new tensor buffer.
//    */
//   SaberStatus re_alloc(Shape shape, DataType type = AK_INVALID) {
//     // if (!shape.dims() == TensorAPI::layout_dims::value) {
//     //     return SaberInvalidValue;
//     // }
//     // if (_is_subbuf || _is_shared) {
//     //     return SaberOutOfAuthority;
//     // }
//     CHECK_EQ(_is_shared || _is_subbuf, false)
//         << "shared tensor could not re_alloc";
//     if (type != AK_INVALID) {
//       _dtype = type;
//       _buf_dtype = type;
//     }
//     _type_len = type_length(type);
//     _shape = shape;
//     _valid_shape = _shape;
//     _offset = Shape::zero(_shape);
//     _buf->alloc(_shape.count() * _type_len);
//     return SaberSuccess;
//   }

//   /**
//    *  \brief Change tensor shape,
//    *  if input shape's count is bigger than the capacity of buffer, alloc a
//    new
//    * buffer.
//    */
//   SaberStatus reshape(Shape valid_shape, Shape shape = Shape(),
//                       Shape offset = Shape()) {
//     if (shape.dims() > 0) {
//       _shape = shape;
//     }
//     if (offset.dims() > 0 && _is_subbuf) {
//       _offset = offset;
//     }
//     CHECK_EQ(valid_shape > Shape::zero(valid_shape), true)
//         << "valid_shape size should > 0";
//     _valid_shape = valid_shape;

//     if (!_is_subbuf) {
//       if (_shape.count() < _valid_shape.count()) {
//         _shape = _valid_shape;
//       }
//       _offset = Shape::zero(_valid_shape);
//     } else {
//       if (_shape == Shape::zero(valid_shape)) {
//         _shape = valid_shape;
//       }
//       //if (!(_valid_shape + _offset <= _shape)) { \
//                 return SaberInvalidValue; \
//             }
//       CHECK_EQ(_valid_shape + _offset <= _shape, true)
//           << "valid_shape + offet should <= shape";
//     }
//     bool exceed_flag = _shape.count() * _type_len > _buf->get_capacity() &&
//                        (_is_subbuf || _is_shared);
//     // if (exceed_flag) {
//     //     return SaberOutOfAuthority;
//     // }
//     CHECK_EQ(exceed_flag, false)
//         << "shared tensor shape exceed origin data buffer size";
//     SABER_CHECK(_buf->re_alloc(_shape.count() * _type_len));
//     return SaberSuccess;
//   }

//   bool is_continue_mem() const {
//     if (!_is_subbuf) {
//       return true;
//     }
//     return _valid_shape.is_continue(_shape);
//   }

//   size_t capacity() const { return _buf->get_capacity(); }

//   /**
//    *  \brief Return shape count, from start index to end index(end index is
//    * excluded). \param start Input start index. \param end   Input end index
//    * (exclude in calculation). \return the size from start index to end
//    index.
//    */
//   long long count(int start, int end) const { return _shape.count(start,
//   end); }

//   /**
//    *  \brief return valid_shape count, from start index to end index(end
//    index
//    * is excluded). \param start input start index. \param end   input end
//    index
//    * (exclude in calculation). \return the size from start index to end
//    index.
//    */
//   long long count_valid(int start, int end) const {
//     return _valid_shape.count(start, end);
//   }

//   /**
//    *  \brief Return tensor shape size, not the valid shape size.
//    */
//   long long size() const { return _shape.count(); }

//   /**
//    *  \brief Return the valid shape size.
//    *  \return Return the valid shape size.
//    */
//   long long valid_size() const { return _valid_shape.count(); }

//   /**
//    *  \brief Return tensor shape dims.
//    */
//   int dims() const { return _valid_shape.dims(); }

//   /**
//    *  \brief compute data stride.
//    */
//   Shape get_stride() const {
//     if (_is_subbuf) {
//       return _shape.get_stride();
//     }
//     return _valid_shape.get_stride();
//   }

//   /**
//    *  \brief Return tensor offset, which holds the offset in each dim.
//    */
//   Shape offset() const { return _offset; }

//   /**
//    *  \brief Return valid shape of tensor
//    */
//   int data_offset() const { return start_index(); }

//   /**
//    * \brief get sequence offset, lot tensor
//    * @return
//    */
//   std::vector<std::vector<int>> get_seq_offset() const { return _seq_offset;
//   }

//   /**
//    * \brief set sequence offset, lot tensor
//    * @param seq_offset
//    * @return
//    */
//   SaberStatus set_seq_offset(std::vector<std::vector<int>> seq_offset) {
//     _seq_offset = seq_offset;
//     return SaberSuccess;
//   }

//   //    /**
//   //     *  \brief Return reference shared_ptr of tensor.
//   //     */
//   //     const std::shared_ptr<Buffer<TargetType>>& get_buf() const {
//   //         return _fbuf;
//   //     }
//   //
//   //    /**
//   //     *  \brief Return reference shared_ptr of tensor.
//   //     */
//   //    const std::shared_ptr<Buffer<TargetType>>& get_lpbuf() const {
//   //        return _lpbuf;
//   //    }

//   /**
//    *  \brief Return tensor device id.
//    */
//   int device_id() const { return _buf->get_id(); }

//   /**
//    *  \brief Return number
//    */
//   int num() const { return _valid_shape.num(); }

//   /**
//    *  \brief Return number index in shape.
//    */
//   int num_index() const { return _valid_shape.num_index(); }

//   /**
//    *  \brief set number to valid shape.
//    */
//   void set_num(int num) {
//     _valid_shape.set_num(num);
//     if (_shape.count() < _valid_shape.count()) {
//       _shape = _valid_shape;
//     }
//   }

//   /**
//    *  \brief Return tensor mutable data pointer void*.
//    */
//   BaseDtype mutable_data() {
//     // synchronize the events tree
//     // sync();
//     CHECK_EQ(device_id(), API::get_device_id())
//         << "tensor is not declared in current device";
//     if (_buf->get_capacity() == 0) {
//       if (std::is_same<TargetType, MLU>::value) {
//         reshape(_valid_shape);
//       } else {
//         return nullptr;
//       }
//     }
//     return static_cast<BaseDtype>(_buf->get_data_mutable());
//   }

//   /**
//    *  \brief Return tensor data pointer, with data type of current tensor
//    * (Dtype*).
//    */
//   const BaseDtype data() const {
//     // synchronize the events tree
//     // sync();
//     CHECK_EQ(device_id(), API::get_device_id())
//         << "tensor is not declared in current device";
//     if (_buf->get_capacity() == 0) {
//       return nullptr;
//     }
//     return static_cast<BaseDtype>(_buf->get_data_mutable());
//   }

//   /**
//    *  \brief Share from same layout_type and same date type tensor,
//    *  if shared tensor target is the same with current tensor target, buffer
//    is
//    * shared; otherwise, tensor buffer is deep copied. only shared buffer ptr,
//    * current tensor will have continuous memory, only if current shape and
//    valid
//    * shape are the same, and offset is all set to 0.
//    */
//   SaberStatus share_from(const Tensor& tensor) {
//     // CHECK_LE(size()*get_dtype_size(),
//     tensor.size()*tensor.get_dtype_size())
//     // << "current tensor size should <= input tensor size";

//     //_is_shared = BufferMemShare(_buf, tensor.get_buf()) > 0;

//     CHECK_GE(tensor._buf->get_capacity(), _shape.count() * _type_len)
//         << "capacity of input tensor should > current tensor";

//     _buf = tensor._buf;
//     _is_subbuf = false;
//     _seq_offset = tensor._seq_offset;
//     _is_shared = true;

//     // if(shared){
//     //     _is_root = false;
//     //     tensor.add_events((EventsTree<TargetType_t>*)(&_events_tree));
//     // } else{
//     //     _is_root = true;
//     // }
//     return SaberSuccess;
//   }

//   SaberStatus share_sub_buffer(const Tensor& tensor, Shape valid_shape,
//                                Shape offset) {
//     //if (valid_shape.dims() != TensorAPI::layout_dims::value \
//             || offset.dims() != TensorAPI::layout_dims::value || \
//             !((offset + valid_shape) <= tensor.shape())) { \
//             return SaberInvalidValue; \
//         }
//     CHECK_EQ(true, (offset + valid_shape) <= tensor.shape())
//         << "offset + valid_shape <= shape";
//     _valid_shape = valid_shape;
//     _offset = offset;
//     _shape = tensor.shape();
//     _buf = tensor._buf;
//     _is_subbuf = true;
//     _is_shared = true;
//     _seq_offset = tensor._seq_offset;
//     return SaberSuccess;
//   }

//   /**
//    *  \brief Deep copy data within region of interest from input tensor.
//    */
//   template <typename TargetType_t>
//   SaberStatus copy_from(const Tensor<TargetType_t>& tensor) {
//     //if (valid_size() != tensor.valid_size()) { \
//             return SaberInvalidValue; \
//         }
//     CHECK_EQ(tensor.get_dtype(), _dtype) << "data type should be the same";
//     CHECK_EQ(valid_size(), tensor.valid_size())
//         << "sizes of two valid shapes must be the same";

//     if (_buf->get_capacity() == 0) {
//       reshape(_valid_shape);
//     }

//     /// get the proper process target wrapper
//     typedef TargetWrapper<TargetType_t> API_t;
//     typedef typename TargetTypeTraits<TargetType_t>::target_type
//     target_type_t; typedef typename IF<std::is_same<target_type,
//     target_type_t>::value, __HtoH,
//                         __DtoH>::Type then_type;
//     typedef typename IF<std::is_same<target_type, target_type_t>::value,
//     __DtoD,
//                         __HtoD>::Type else_type;
//     typedef typename IF<std::is_same<target_category, __host_target>::value,
//                         then_type, else_type>::Type flag_type;
//     typedef typename IF<std::is_same<target_category, __host_target>::value,
//                         API_t, API>::Type process_API;

//     typedef typename DataTraitBase<TargetType_t>::PtrDtype BaseDtype_src;

//     /// both tensors are continuous, copy entire buffer
//     if (is_continue_mem() && tensor.is_continue_mem()) {
//       int dst_data_offset = data_offset();
//       int src_data_offset = tensor.data_offset();

//       BaseDtype ptr_dst = _buf->get_data_mutable();
//       const BaseDtype_src ptr_src = tensor.data();

//       process_API::sync_memcpy(ptr_dst, _type_len * dst_data_offset,
//                                device_id(), ptr_src,
//                                _type_len * src_data_offset,
//                                tensor.device_id(), _type_len * valid_size(),
//                                flag_type());

//       return SaberSuccess;
//     }

//     Shape sh_dst = _shape;
//     Shape val_sh_dst = _valid_shape;
//     Shape sh_src = tensor.shape();
//     Shape val_sh_src = tensor.valid_shape();
//     // Shape off_dst = _offset;
//     // Shape off_src = tensor.offset();

//     if (is_continue_mem()) {
//       sh_dst = _valid_shape;
//     }
//     if (tensor.is_continue_mem()) {
//       sh_src = val_sh_src;
//     }

//     int dim_dst = dims();
//     int dim_src = tensor.dims();

//     /// check the beginning axis of dis_continue memory
//     int axis_discontinue_dst = -1;
//     int axis_discontinue_src = -1;
//     for (int i = dim_dst - 1; i >= 0; i--) {
//       if (val_sh_dst[i] == sh_dst[i]) {
//         continue;
//       } else {
//         axis_discontinue_dst = i;
//         break;
//       }
//     }
//     for (int i = dim_src - 1; i >= 0; i--) {
//       if (val_sh_src[i] == sh_src[i]) {
//         continue;
//       } else {
//         axis_discontinue_src = i;
//         break;
//       }
//     }
//     // printf("dst axis=%d, src axis=%d\n", axis_discontinue_dst,
//     // axis_discontinue_src);

//     /// only copy the region of interest
//     /// compute the copy length of each memcpy
//     int cpy_len_dst = 1;
//     int cpy_len_src = 1;
//     if (axis_discontinue_dst < 0) {
//       cpy_len_dst = valid_size();
//     } else {
//       for (int i = axis_discontinue_dst; i < dim_dst; i++) {
//         cpy_len_dst *= val_sh_dst[i];
//       }
//     }
//     if (axis_discontinue_src < 0) {
//       cpy_len_src = tensor.valid_size();
//     } else {
//       for (int i = axis_discontinue_src; i < dim_src; i++) {
//         cpy_len_src *= val_sh_src[i];
//       }
//     }
//     // printf("cpy_len_dst=%d, %d, cpy_len_src=%d, %d\n", cpy_len_dst,
//     // valid_size(), cpy_len_src, tensor.valid_size());
//     int cpy_len = cpy_len_dst < cpy_len_src ? cpy_len_dst : cpy_len_src;

//     /// compute the total copy times
//     int cpy_num = valid_size() / cpy_len;
//     // printf("cpy_len=%d, cpy_num=%d\n", cpy_len, cpy_num);

//     /// compute the stride and start index of dst buffer and src buffer
//     std::vector<int> count_dst(abs(axis_discontinue_dst) + 1);
//     std::vector<int> count_src(abs(axis_discontinue_src) + 1);

//     Shape stride_dst = get_stride();
//     Shape stride_src = tensor.get_stride();

//     count_dst[abs(axis_discontinue_dst)] =
//         count_src[abs(axis_discontinue_src)] = 1;
//     for (int i = axis_discontinue_dst - 1; i >= 0; i--) {
//       if (i == axis_discontinue_dst - 1) {
//         count_dst[i] = 1;
//       } else {
//         count_dst[i] = val_sh_dst[i + 1] * count_dst[i + 1];
//       }
//     }
//     for (int i = axis_discontinue_src - 1; i >= 0; i--) {
//       if (i == axis_discontinue_src - 1) {
//         count_src[i] = 1;
//       } else {
//         count_src[i] = val_sh_src[i + 1] * count_src[i + 1];
//       }
//     }

//     /// compute the start position of each buffer, memcpy from src to dst
//     int ratio_dst = cpy_len_dst / cpy_len;
//     int ratio_src = cpy_len_src / cpy_len;

//     int dst_data_offset = data_offset();
//     int src_data_offset = tensor.data_offset();

//     BaseDtype ptr_dst = _buf->get_data_mutable();
//     const BaseDtype_src ptr_src = tensor.data();

//     for (int i = 0; i < cpy_num; ++i) {
//       int idx_dst =
//           (i % ratio_dst) * cpy_len;  //off_dst[abs(axis_discontinue_dst)] *
//           \
//                 stride_dst[abs(axis_discontinue_dst)] + (i % ratio_dst) *
//                 cpy_len;
//       int res_dst = i / ratio_dst;
//       for (int j = 0; j < axis_discontinue_dst; ++j) {
//         int div = res_dst / count_dst[j];
//         idx_dst += (div /*+ off_dst[j]*/) * stride_dst[j];
//         res_dst = res_dst % count_dst[j];
//       }
//       int idx_src =
//           (i % ratio_src) * cpy_len;  //off_src[abs(axis_discontinue_src)] *
//           \
//                 stride_src[abs(axis_discontinue_src)] + (i % ratio_src) *
//                 cpy_len;
//       int res_src = i / ratio_src;
//       for (int j = 0; j < axis_discontinue_src; ++j) {
//         int div = res_src / count_src[j];
//         idx_src += (div /*+ off_src[j]*/) * stride_src[j];
//         res_src = res_src % count_src[j];
//       }
//       // printf("i: %d, idx_src: %d, idx_dst: %d\n", i, idx_src, idx_dst);

//       int cpy_dst_offset = dst_data_offset + idx_dst;
//       int cpy_src_offset = src_data_offset + idx_src;

//       process_API::sync_memcpy(ptr_dst, _type_len * cpy_dst_offset,
//       device_id(),
//                                ptr_src, _type_len * cpy_src_offset,
//                                tensor.device_id(), _type_len * cpy_len,
//                                flag_type());
//     }
//     return SaberSuccess;
//   }

//   /**
//    * \brief Asynchronously copy entire buffer from source tensor.
//    */
//   template <
//       typename TargetType_t,
//       typename stream_type = typename IF<
//           std::is_same<typename
//           TargetTypeTraits<TargetType>::target_category,
//                        __host_target>::value,
//           typename TargetWrapper<TargetType_t>::stream_t,
//           typename TargetWrapper<TargetType>::stream_t>::Type>
//   SaberStatus async_copy_from(const Tensor<TargetType_t>& tensor,
//                               stream_type stream) {
//     CHECK_EQ(tensor.get_dtype(), _dtype) << "data type should be the same";
//     CHECK_EQ(valid_size(), tensor.valid_size())
//         << "sizes of two valid shapes must be the same";

//     if (_buf->get_capacity() == 0) {
//       reshape(_valid_shape);
//     }

//     /// get the proper process target wrapper
//     typedef TargetWrapper<TargetType_t> API_t;
//     typedef typename TargetTypeTraits<TargetType_t>::target_type
//     target_type_t; typedef typename IF<std::is_same<target_type,
//     target_type_t>::value, __HtoH,
//                         __DtoH>::Type then_type;
//     typedef typename IF<std::is_same<target_type, target_type_t>::value,
//     __DtoD,
//                         __HtoD>::Type else_type;
//     typedef typename IF<std::is_same<target_category, __host_target>::value,
//                         then_type, else_type>::Type flag_type;
//     typedef typename IF<std::is_same<target_category, __host_target>::value,
//                         API_t, API>::Type process_API;

//     typedef typename DataTraitBase<TargetType>::PtrDtype BaseDtype_src;

//     /// return if src and dst data ptrs are the same
//     if (std::is_same<TargetType, TargetType_t>::value) {
//       if ((const void*)data() == (const void*)(tensor.data())) {
//         return SaberSuccess;
//       }
//     }

//     /// both tensors are continuous, copy entire buffer
//     if (is_continue_mem() && tensor.is_continue_mem()) {
//       int dst_data_offset = data_offset();
//       int src_data_offset = tensor.data_offset();

//       BaseDtype ptr_dst = _buf->get_data_mutable();
//       const BaseDtype_src ptr_src = tensor.data();

//       process_API::async_memcpy(ptr_dst, _type_len * dst_data_offset,
//                                 device_id(), ptr_src,
//                                 _type_len * src_data_offset,
//                                 tensor.device_id(), _type_len * valid_size(),
//                                 stream, flag_type());

//       return SaberSuccess;
//     }

//     Shape sh_dst = _shape;
//     Shape val_sh_dst = _valid_shape;
//     Shape sh_src = tensor.shape();
//     Shape val_sh_src = tensor.valid_shape();
//     // Shape off_dst = _offset;
//     // Shape off_src = tensor.offset();

//     if (is_continue_mem()) {
//       sh_dst = _valid_shape;
//     }
//     if (tensor.is_continue_mem()) {
//       sh_src = val_sh_src;
//     }

//     int dim_dst = dims();
//     int dim_src = tensor.dims();

//     /// check the beginning axis of dis_continue memory
//     int axis_discontinue_dst = -1;
//     int axis_discontinue_src = -1;
//     for (int i = dim_dst - 1; i >= 0; i--) {
//       if (val_sh_dst[i] == sh_dst[i]) {
//         continue;
//       } else {
//         axis_discontinue_dst = i;
//         break;
//       }
//     }
//     for (int i = dim_src - 1; i >= 0; i--) {
//       if (val_sh_src[i] == sh_src[i]) {
//         continue;
//       } else {
//         axis_discontinue_src = i;
//         break;
//       }
//     }
//     // printf("dst axis=%d, src axis=%d\n", axis_discontinue_dst,
//     // axis_discontinue_src);

//     /// only copy the region of interest
//     /// compute the copy length of each memcpy
//     int cpy_len_dst = 1;
//     int cpy_len_src = 1;
//     if (axis_discontinue_dst < 0) {
//       cpy_len_dst = valid_size();
//     } else {
//       for (int i = axis_discontinue_dst; i < dim_dst; i++) {
//         cpy_len_dst *= val_sh_dst[i];
//       }
//     }
//     if (axis_discontinue_src < 0) {
//       cpy_len_src = tensor.valid_size();
//     } else {
//       for (int i = axis_discontinue_src; i < dim_src; i++) {
//         cpy_len_src *= val_sh_src[i];
//       }
//     }
//     // printf("cpy_len_dst=%d, %d, cpy_len_src=%d, %d\n", cpy_len_dst,
//     // valid_size(), cpy_len_src, tensor.valid_size());
//     int cpy_len = cpy_len_dst < cpy_len_src ? cpy_len_dst : cpy_len_src;

//     /// compute the total copy times
//     int cpy_num = valid_size() / cpy_len;
//     // printf("cpy_len=%d, cpy_num=%d\n", cpy_len, cpy_num);

//     /// compute the stride and start index of dst buffer and src buffer
//     std::vector<int> count_dst(abs(axis_discontinue_dst) + 1);
//     std::vector<int> count_src(abs(axis_discontinue_src) + 1);

//     Shape stride_dst = get_stride();
//     Shape stride_src = tensor.get_stride();

//     count_dst[abs(axis_discontinue_dst)] =
//         count_src[abs(axis_discontinue_src)] = 1;
//     for (int i = axis_discontinue_dst - 1; i >= 0; i--) {
//       if (i == axis_discontinue_dst - 1) {
//         count_dst[i] = 1;
//       } else {
//         count_dst[i] = val_sh_dst[i + 1] * count_dst[i + 1];
//       }
//     }
//     for (int i = axis_discontinue_src - 1; i >= 0; i--) {
//       if (i == axis_discontinue_src - 1) {
//         count_src[i] = 1;
//       } else {
//         count_src[i] = val_sh_src[i + 1] * count_src[i + 1];
//       }
//     }

//     /// compute the start position of each buffer, memcpy from src to dst
//     int ratio_dst = cpy_len_dst / cpy_len;
//     int ratio_src = cpy_len_src / cpy_len;

//     int dst_data_offset = data_offset();
//     int src_data_offset = tensor.data_offset();

//     BaseDtype ptr_dst = _buf->get_data_mutable();
//     const BaseDtype_src ptr_src = tensor.data();

//     for (int i = 0; i < cpy_num; ++i) {
//       int idx_dst =
//           (i % ratio_dst) * cpy_len;  //off_dst[abs(axis_discontinue_dst)] *
//           \
//                 stride_dst[abs(axis_discontinue_dst)] + (i % ratio_dst) *
//                 cpy_len;
//       int res_dst = i / ratio_dst;
//       for (int j = 0; j < axis_discontinue_dst; ++j) {
//         int div = res_dst / count_dst[j];
//         idx_dst += (div /*+ off_dst[j]*/) * stride_dst[j];
//         res_dst = res_dst % count_dst[j];
//       }
//       int idx_src =
//           (i % ratio_src) * cpy_len;  //off_src[abs(axis_discontinue_src)] *
//           \
//                 stride_src[abs(axis_discontinue_src)] + (i % ratio_src) *
//                 cpy_len;
//       int res_src = i / ratio_src;
//       for (int j = 0; j < axis_discontinue_src; ++j) {
//         int div = res_src / count_src[j];
//         idx_src += (div /*+ off_src[j]*/) * stride_src[j];
//         res_src = res_src % count_src[j];
//       }
//       // printf("i: %d, idx_src: %d, idx_dst: %d\n", i, idx_src, idx_dst);

//       int cpy_dst_offset = dst_data_offset + idx_dst;
//       int cpy_src_offset = src_data_offset + idx_src;

//       process_API::async_memcpy(ptr_dst, _type_len * cpy_dst_offset,
//                                 device_id(), ptr_src,
//                                 _type_len * cpy_src_offset,
//                                 tensor.device_id(), _type_len * cpy_len,
//                                 stream, flag_type());
//     }
//     return SaberSuccess;
//   }

//   /**
//    *  \brief Add events when tensor is shared to others.
//    */
//   void add_events(EventsTree<TargetType>* events) {
//     _events_tree.insert_children(events);
//   }

//   /**
//    *  \brief Synchronize the event tree, wait util all events are done.
//    */
//   void sync() { _events_tree.sync_tree(); }

//   /**
//    *  \brief record Event to current tensor.
//    *  \param stream  Input processing stream.
//    */
//   void record_event(typename API::stream_t stream) {
//     _events_tree._events.record(stream);
//   }

//   bool get_posstive_flag() { return _is_all_positive; }

//   void set_posstive_flag(bool is_all_posstive) {
//     _is_all_positive = is_all_posstive;
//   }

//  private:
//   //! scale for quantization
//   std::vector<float> _scale;
//   bool _is_all_positive{false};

//   ///< Length of datatype.
//   DataType _dtype{AK_FLOAT};
//   size_t _type_len{4};
//   DataType _buf_dtype{AK_FLOAT};

//   ///< Represent the raw mem shape.
//   Shape _shape;
//   ///< Represent the mem you have right to access shape.
//   Shape _valid_shape;
//   ///< Represent the offset idx between _shape and _real_shape.
//   Shape _offset;
//   ///< Buffer shared ptr, hold the data pointer, and buffer capacity.
//   std::shared_ptr<Buffer<TargetType>> _buf{nullptr};
//   Buffer<TargetType> _scale_buf;
//   ///< Events tree, to synchronize the tensor.
//   EventsTree<TargetType> _events_tree;
//   ///< share sub-buffer flag.
//   bool _is_subbuf{false};
//   bool _is_shared{false};

//   //! lot tensor
//   std::vector<std::vector<int>> _seq_offset;

//   /// Get data real start index.
//   int start_index() const {
//     if (!_is_subbuf) {
//       return 0;
//     }
//     Shape stride = get_stride();
//     int idx = 0;
//     for (int i = 0; i < stride.size(); ++i) {
//       idx += _offset[i] * stride[i];
//     }
//     return idx;
//   }
// };

}  // namespace device
}  // namespace ace

#pragma once

#include <string>
#include <type_traits>
#include <utility>

#include "../framework/core/status.h"
#include "ir/types_generated.h"

namespace ace {
namespace device {

/// Type changing for  std::vector<bool> which considered a mistake in STL.
template <typename T>
struct std_vector_type_warpper {
  typedef T type;
  typedef T ret_type;
};

template <>
struct std_vector_type_warpper<bool> {
  typedef std::string type;
  typedef bool ret_type;
};

template <>
struct std_vector_type_warpper<const bool> {
  typedef std::string type;
  typedef const bool ret_type;
};

template <typename T>
struct is_bool_type : std::is_same<T, bool> {};

template <typename T>
struct is_bool_type<const T> : std::is_same<T, bool> {};

template <bool Boolean>
struct Bool2Type {};

template <size_t index, typename Arg, typename... Args>
struct ParamPackType;

template <size_t index, typename Arg, typename... Args>
struct ParamPackType : ParamPackType<index - 1, Args...> {};

template <typename Arg, typename... Args>
struct ParamPackType<0, Arg, Args...> {
  typedef Arg type;
};

template <typename T>
struct function_traits;

template <typename RetType, typename... Args>
struct function_traits<RetType(Args...)> {
  typedef RetType return_type;
  enum { size = sizeof...(Args) };

  template <size_t index>
  struct Param {
    typedef typename ParamPackType<index, Args...>::type type;
  };
};

template <typename ClassType, typename RetType, typename... Args>
struct function_traits<RetType (ClassType::*)(Args...) const> {
  typedef RetType return_type;
  enum { size = sizeof...(Args) };

  template <size_t index>
  struct Param {
    typedef typename ParamPackType<index, Args...>::type type;
  };
};

template <typename LambdaFunc>
struct function_traits : function_traits<decltype(&LambdaFunc::operator())> {};

template <typename RetType, typename... Args>
struct function_traits<RetType(Args...) const>
    : function_traits<RetType(Args...)> {};

/// Judge if the function return type is void.
template <typename>
struct is_void_function;

template <typename functor>
struct is_void_function
    : std::is_void<typename function_traits<functor>::return_type> {};

/// Judge if the function return type is Status.
template <typename>
struct is_status_function;

template <typename functor>
struct is_status_function
    : std::is_same<typename function_traits<functor>::return_type, Status> {};

struct __invalid_type {};

template <typename Ttype, DataType datatype>
struct DataTrait {
  typedef __invalid_type Dtype;
  typedef __invalid_type PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, DataType_INT_16> {
  typedef short Dtype;
  typedef short* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, DataType_FLOAT_32> {
  typedef float Dtype;
  typedef float* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, DataType_INT_8> {
  typedef char Dtype;
  typedef char* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, DataType_INT_32> {
  typedef int Dtype;
  typedef int* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, DataType_INT_64> {
  typedef long Dtype;
  typedef long* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, DataType_UINT_8> {
  typedef unsigned char Dtype;
  typedef unsigned char* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, DataType_UINT_16> {
  typedef unsigned short Dtype;
  typedef unsigned short* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, DataType_UINT_32> {
  typedef unsigned int Dtype;
  typedef unsigned int* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, DataType_UINT_64> {
  typedef unsigned int Dtype;
  typedef unsigned int* PtrDtype;
};

}  // namespace device
}  // namespace ace
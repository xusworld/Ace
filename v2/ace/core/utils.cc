#include <numeric>

#include "utils.h"

namespace ace {

int DataType2Bytes(const DataType dtype) {
  switch (dtype) {
    case DataType_NONE:
      return 0;
    case DataType_INT_8:
      return 1;
    case DataType_INT_16:
      return 2;
    case DataType_INT_32:
      return 4;
    case DataType_INT_64:
      return 8;
    case DataType_UINT_8:
      return 1;
    case DataType_UINT_16:
      return 2;
    case DataType_UINT_32:
      return 4;
    case DataType_UINT_64:
      return 8;
    case DataType_FLOAT_16:
      return 2;
    case DataType_FLOAT_32:
      return 4;
    case DataType_FLOAT_64:
      return 8;
      break;
  }

  return 0;
}

std::string DataType2String(const DataType dtype) {
  switch (dtype) {
    case DataType_NONE:
      return "none";
    case DataType_INT_8:
      return "int8";
    case DataType_INT_16:
      return "int16";
    case DataType_INT_32:
      return "int32";
    case DataType_INT_64:
      return "int64";
    case DataType_UINT_8:
      return "uint8";
    case DataType_UINT_16:
      return "uint16";
    case DataType_UINT_32:
      return "uint32";
    case DataType_UINT_64:
      return "uint64";
    case DataType_FLOAT_16:
      return "half";
    case DataType_FLOAT_32:
      return "float";
    case DataType_FLOAT_64:
      return "double";
      break;
  }

  return 0;
}

int32_t GetBufferBytes(const DataType dtype, const int size) {
  return size * DataType2Bytes(dtype);
}

int32_t GetBufferBytes(const DataType dtype, const std::vector<int32_t>& dims) {
  const int32_t size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  return size * DataType2Bytes(dtype);
}

int32_t GetBufferBytes(const DataType dtype, const TensorShape& shape) {
  const int32_t size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  return size * DataType2Bytes(dtype);
}

}  // namespace ace
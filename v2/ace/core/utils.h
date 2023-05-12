#pragma once

#include "ir/types_generated.h"
#include "tensor_shape.h"

namespace ace {

// DataType to bytes.
inline int DataType2Bytes(const DataType dtype);

// DataType to string.
inline std::string DataType2String(const DataType dtype);

int32_t GetBufferBytes(const DataType dtype, const int size);
int32_t GetBufferBytes(const DataType dtype, const std::vector<int32_t>& dims);
int32_t GetBufferBytes(const DataType dtype, const TensorShape& shape);

}  // namespace ace
#pragma once

#include <cstdint>

#include "ir/types_generated.h"
#include "tensor_shape.h"

namespace ace {
namespace device {

int32_t GetBufferBytes(const DataType dtype, const int size);
int32_t GetBufferBytes(const DataType dtype, const std::vector<int32_t>& dims);
int32_t GetBufferBytes(const DataType dtype, const TensorShape& shape);

}  // namespace device
}  // namespace ace
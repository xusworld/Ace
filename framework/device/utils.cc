
#include <functional>
#include <numeric>

#include "../core/utils.h"
#include "utils.h"

namespace ace {
namespace device {

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

}  // namespace device
}  // namespace ace
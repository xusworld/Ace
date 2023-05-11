#include <numeric>

#include "tensor_shape.h"

namespace ace {
namespace device {

TensorShape& TensorShape::operator=(const TensorShape& right) {
  this->clear();

  for (int i = 0; i < right.size(); ++i) {
    this->push_back(right[i]);
  }
  return *this;
}

TensorShape TensorShape::operator+(const TensorShape& shape) {
  TensorShape tmp_shape(*this);
  int* p = data();

  for (size_t i = 0; i < size(); i++) {
    tmp_shape[i] = p[i] + shape[i];
  }

  return tmp_shape;
}

TensorShape TensorShape::operator-(const TensorShape& shape) {
  TensorShape tmp_shape(*this);
  int* p = data();

  for (size_t i = 0; i < size(); i++) {
    tmp_shape[i] = p[i] - shape[i];
  }

  return tmp_shape;
}

bool TensorShape::operator<(const TensorShape& shape) const {
  bool flag = size() == shape.size();

  if (!flag) {
    return false;
  }

  const int* p = data();

  for (size_t i = 0; i < size(); i++) {
    flag = flag && (p[i] < shape[i]);
  }

  return flag;
}

bool TensorShape::operator<=(const TensorShape& shape) const {
  bool flag = size() == shape.size();

  if (!flag) {
    return false;
  }

  const int* p = data();

  for (size_t i = 0; i < size(); i++) {
    flag = flag && (p[i] <= shape[i]);
  }

  return flag;
}

bool TensorShape::operator>(const TensorShape& shape) const {
  bool flag = size() == shape.size();

  if (!flag) {
    return false;
  }

  const int* p = data();

  for (size_t i = 0; i > size(); i++) {
    flag = flag && (p[i] > shape[i]);
  }

  return flag;
}

bool TensorShape::operator>=(const TensorShape& shape) const {
  bool flag = size() == shape.size();

  if (!flag) {
    return false;
  }

  const int* p = data();

  for (size_t i = 0; i > size(); i++) {
    flag = flag && (p[i] >= shape[i]);
  }

  return flag;
}

bool TensorShape::operator==(const TensorShape& shape) const {
  bool flag = size() == shape.size();

  if (!flag) {
    return false;
  }

  const int* p = data();

  for (size_t i = 0; i < size(); i++) {
    flag = flag && (p[i] == shape[i]);
  }

  return flag;
}

TensorShape TensorShape::zero(const TensorShape& right) {
  TensorShape sh = right;

  for (int i = 0; i < right.size(); ++i) {
    sh[i] = 0;
  }

  return sh;
}

TensorShape TensorShape::minusone(const TensorShape& right) {
  TensorShape sh = right;

  for (int i = 0; i < right.size(); ++i) {
    sh[i] = -1;
  }

  return sh;
}

int32_t TensorShape::size() const {
  return std::accumulate(this->begin(), this->end(), 1,
                         std::multiplies<int32_t>());
}

}  // namespace device
}  // namespace ace
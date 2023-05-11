#pragma once

#include <glog/logging.h>

#include <cstdint>
#include <iostream>
#include <iterator>
#include <vector>

#include "ir/types_generated.h"

namespace ace {
namespace device {

namespace {
using Vector = std::vector<int32_t>;
}

class TensorShape : public Vector {
 public:
  TensorShape() : Vector() {}
  TensorShape(const TensorShape& shape) {
    this->clear();

    for (int i = 0; i < shape.size(); ++i) {
      this->push_back(shape[i]);
    }
  }

  ~TensorShape() = default;

  TensorShape& operator=(const TensorShape& right);
  TensorShape operator+(const TensorShape& shape);
  TensorShape operator-(const TensorShape& shape);

  bool operator<(const TensorShape& shape) const;
  bool operator<=(const TensorShape& shape) const;
  bool operator>(const TensorShape& shape) const;
  bool operator>=(const TensorShape& shape) const;
  bool operator==(const TensorShape& shape) const;

  int32_t size() const;

  static TensorShape zero(const TensorShape& right);
  static TensorShape minusone(const TensorShape& right);
};

}  // namespace device
}  // namespace ace

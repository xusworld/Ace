#pragma once

#include "framework/device/operator.h"
#include "ir/types_generated.h"

namespace ace {
namespace device {

template <typename DeviceType, DataType dtype>
class OperatorImpl {
 public:
  OperatorImpl() = default;
  virtual ~OperatorImpl() = default;

  virtual Status init(const OpParam &param, std::vector<Tensor *> inputs,
                      std::vector<Tensor *> outputs) = 0;

  virtual Status create(const OpParam &param, std::vector<Tensor *> inputs,
                        std::vector<Tensor *> outputs) = 0;

  virtual Status dispatch(const OpParam &param, std::vector<Tensor *> inputs,
                          std::vector<Tensor *> outputs) = 0;

 protected:
  std::string name_;
};

}  // namespace device
}  // namespace ace
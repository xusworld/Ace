#pragma once

#include "../operator_impl.h"
#include "ir/types_generated.h"
#include "status.h"

namespace ace {
namespace device {

template <typename DeviceType, DataType dtype>
class ElementwiseImpl : public OperatorImpl<DeviceType, dtype> {
  virtual Status init(const OpParam &param, std::vector<Tensor *> inputs,
                      std::vector<Tensor *> outputs) override;

  virtual Status create(const OpParam &param, std::vector<Tensor *> inputs,
                        std::vector<Tensor *> outputs) override;

  virtual Status dispatch(const OpParam &param, std::vector<Tensor *> inputs,
                          std::vector<Tensor *> outputs) override;
};

}  // namespace device
}  // namespace ace
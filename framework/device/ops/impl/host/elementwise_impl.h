#pragma once

#include "../operator_impl.h"
#include "ir/types_generated.h"

namespace ace {
namespace device {

template <typename DeviceType, DataType dtype>
class ElementwiseImpl : public OperatorImpl<DeviceType, dtype> {};

}  // namespace device
}  // namespace ace
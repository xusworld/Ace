#pragma once

#include "../operator.h"
#include "ir/op_option_generated.h"
#include "ir/types_generated.h"

namespace ace {
namespace device {

/********************
Elementwise operatos:
  Abs,
  BoundedRelu,
  Clip,
  ClipV2,
  ClippedRelu,
  Elu,
  Exp,
  GeluTanh,
  HardSigmoid,
  HardSwish,
  LeakyRelu,
  Linear,
  Log,
  Logistic,
  LogSigmoid,
  Mish,
  Pow,
  PRelu,
  Relu,
  Relu6,
  Round,
  Selu,
  Sigmoid,
  SoftRelu,
  SoftReluV2,
  Sqrt,
  Swish,
  Tanh,
********************/

namespace {

template <typename OpOption_T>
const OpOption_T *GetOpOption(const OpParam &op_param) {
  return static_cast<const OpOption_T *>(op_param.option);
}
}  // namespace

template <typename DeviceType, DataType dtype>
class ElementwiseOperator : public Operator<DeviceType, dtype> {
 public:
  ElementwiseOperator() {}

  virtual Status init(const OpParam &op_param,
                      const std::vector<Tensor *> &inputs,
                      std::vector<Tensor *> outputs) {
    this->inputs_ = inputs;
    this->outputs_ = outputs;
    this->dtype_ = dtype;

    const AbsOptionT *abs_option = op_param.option.AsAbsOption();

    return Status::OK();
  }

  virtual ~ElementwiseOperator() {}

 private:
};

}  // namespace device
}  // namespace ace
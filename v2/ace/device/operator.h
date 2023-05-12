#pragma once

#include "core/status.h"
#include "core/tensor.h"
#include "ir/op_generated.h"
#include "ir/op_option_generated.h"
#include "ir/types_generated.h"

namespace ace {
namespace device {

// OpParam 中持有 operator 的配置信息
struct OpParam {
  OpOptionUnion option;
};

template <typename DeviceType, DataType dtype>
class Operator {
 public:
  Operator();
  virtual ~Operator() = default;

  virtual Status init(const OpParam &op_param,
                      const std::vector<Tensor *> &inputs,
                      std::vector<Tensor *> outputs) {
    inputs_ = inputs;
    outputs_ = outputs;
    dtype_ = dtype;
    return Status::OK();
  }

  virtual Status invoke() = 0;

  virtual Status shape_infer() = 0;

 protected:
  // basic op's attributes
  std::string name_;
  std::string desc_;
  OpType op_type_;
  DataType dtype_;
  OpOption op_option_;

  std::vector<Tensor *> inputs_;
  std::vector<Tensor *> outputs_;
  std::vector<OperatorImpl *> impl_;
};

}  // namespace device
}  // namespace ace
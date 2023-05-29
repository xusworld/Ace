//
//  CPUQuantizedLogistic.cpp
//  MNN
//
//  Created by MNN on 2018/12/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "device/cpu/CPUDevice.h"
#ifdef MNN_SUPPORT_DEPRECATED_OP
#include "core/Macro.h"
#include "device/cpu/CPUFixedPoint.hpp"
#include "device/cpu/CPUQuantizationUtils.hpp"
#include "device/cpu/CPUQuantizedLogistic.hpp"
#include "device/cpu/compute/OptimizedComputer.hpp"

namespace tars {

CPUQuantizedLogistic::CPUQuantizedLogistic(Device *backend, const Op *op)
    : Operation(backend) {
  mLogisticParam = op->main_as_QuantizedLogistic();
}

Status CPUQuantizedLogistic::onResize(const std::vector<Tensor *> &inputs,
                                      const std::vector<Tensor *> &outputs) {
  MNN_ASSERT(1 == inputs.size() && 1 == outputs.size());
  MNN_ASSERT(0 == mLogisticParam->outputQuantizedParam()->zeroPoint() &&
             1. / 256 == mLogisticParam->outputQuantizedParam()->scale());

  static constexpr int kInputIntegerBits = 4;
  const double inputRealMultiplier =
      mLogisticParam->inputQuantizedParam()->scale() *
      static_cast<double>(1 << (31 - kInputIntegerBits));
  QuantizeMultiplierGreaterThanOne(inputRealMultiplier, &mInputMultiplier,
                                   &mInputLeftShift);
  mInputZeroPoint = mLogisticParam->inputQuantizedParam()->zeroPoint();
  mInputRangeRadius = CalculateInputRadius(kInputIntegerBits, mInputLeftShift);
  return Status::OK();
}

Status CPUQuantizedLogistic::onExecute(
    const std::vector<tars::Tensor *> &inputs,
    const std::vector<tars::Tensor *> &outputs) {
  auto input = inputs[0], output = outputs[0];
  std::vector<int> inputDims, outputDims;
  for (int i = 0; i < input->buffer().dimensions; i++) {
    inputDims.push_back(input->buffer().dim[i].extent);
  }
  for (int i = 0; i < output->buffer().dimensions; i++) {
    outputDims.push_back(output->buffer().dim[i].extent);
  }

  Optimized::Logistic(input->host<uint8_t>(), inputDims, mInputZeroPoint,
                      mInputRangeRadius, mInputMultiplier, mInputLeftShift,
                      output->host<uint8_t>(), outputDims);

  return Status::OK();
}

class CPUQuantizedLogisticCreator : public CPUDevice::Creator {
 public:
  virtual Operation *onCreate(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs,
                              const tars::Op *op, Device *backend) const {
    return new CPUQuantizedLogistic(backend, op);
  }
};
}  // namespace tars
#endif
namespace tars {
REGISTER_CPU_OP_CREATOR_OLD(CPUQuantizedLogisticCreator,
                            OpType_QuantizedLogistic);
};

//
//  CPULayerNorm.cpp
//  MNN
//
//  Created by MNN on 2020/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>

#include "MNN_generated.h"
#include "core/Concurrency.h"
#include "core/OpCommonUtils.hpp"
#include "core/operation.h"
#include "device/cpu/CPUDevice.h"
#include "device/cpu/compute/CommonOptFunction.h"

namespace tars {

class CPULayerNorm : public Operation {
 public:
  explicit CPULayerNorm(const tars::Op* op, Device* backend);
  virtual ~CPULayerNorm();

  Status onExecute(const std::vector<Tensor*>& inputs,  // NOLINT
                   const std::vector<Tensor*>& outputs) override;

  Status onResize(const std::vector<Tensor*>& inputs,  // NOLINT
                  const std::vector<Tensor*>& outputs) override;

 private:
  bool allocGammaBeta(int size);

 private:
  int axis_size = 0;
  int inner_size_ = 1;
  int outter_size_ = 1;
  int group_ = 1;
  float epsilon_ = 0.001;

  std::unique_ptr<Tensor> gamma_;
  std::unique_ptr<Tensor> beta_;
  bool has_gamma_beta_ = false;
};

bool CPULayerNorm::allocGammaBeta(int size) {
  has_gamma_beta_ = true;
  gamma_.reset(Tensor::createDevice<float>({size}));
  auto status = backend()->onAcquireBuffer(gamma_.get(), Device::STATIC);
  if (!status) {
    MNN_ERROR("Out of memory when gamma is acquired in CPULayerNorm.\n");
    return false;
  }
  beta_.reset(Tensor::createDevice<float>({size}));
  status = backend()->onAcquireBuffer(beta_.get(), Device::STATIC);
  if (!status) {
    MNN_ERROR("Out of memory when beta is acquired in CPULayerNorm.\n");
    return false;
  }
  return true;
}

CPULayerNorm::CPULayerNorm(const tars::Op* op, Device* backend)
    : Operation(backend) {
  const auto* layer_norm_param = op->main_as_LayerNorm();
  axis_size = layer_norm_param->axis()->size();
  group_ = layer_norm_param->group();
  epsilon_ = layer_norm_param->epsilon();

  if (USE_EXTERNAL_DATA(layer_norm_param)) {
    auto size = layer_norm_param->external()->Get(1);
    allocGammaBeta(size);
    OpCommonUtils::loadExternalDatas(
        backend, {gamma_->host<char>(), beta_->host<char>()},
        layer_norm_param->external()->data());
    return;
  }

  if (layer_norm_param->gamma() && layer_norm_param->beta()) {
    int size = layer_norm_param->gamma()->size();
    if (layer_norm_param->beta()->size() != size) {
      MNN_ERROR("Size of gamma and beta are not match in CPULayerNorm.\n");
    }
    allocGammaBeta(size);
    const float* gamma_data = layer_norm_param->gamma()->data();
    memcpy(gamma_->host<float>(), gamma_data, size * sizeof(float));
    const float* beta_data = layer_norm_param->beta()->data();
    memcpy(beta_->host<float>(), beta_data, size * sizeof(float));
  }
}

Status CPULayerNorm::onExecute(const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) {
  const float* gamma = has_gamma_beta_ ? gamma_->host<float>() : nullptr;
  const float* beta = has_gamma_beta_ ? beta_->host<float>() : nullptr;

  const float* input = inputs.at(0)->host<float>();
  float* output = outputs.at(0)->host<float>();
  MNN_CONCURRENCY_BEGIN(tId, outter_size_) {
    const float* inner_input = input + tId * inner_size_;
    float* inner_output = output + tId * inner_size_;
    MNNNorm(inner_output, inner_input, gamma, beta, epsilon_, inner_size_);
  }
  MNN_CONCURRENCY_END();
  return Status::OK();
}

Status CPULayerNorm::onResize(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs) {
  outter_size_ = 1;
  inner_size_ = 1;
  int rank = inputs.at(0)->dimensions();
  if (group_ > 1) {
    outter_size_ = inputs.at(0)->length(0) * group_;
    for (int i = 1; i < rank; i++) {
      inner_size_ *= inputs.at(0)->length(i);
    }
    inner_size_ /= group_;
    return Status::OK();
  }
  for (int i = 0; i < rank - axis_size; ++i) {
    outter_size_ *= inputs.at(0)->length(i);
  }
  for (int i = rank - axis_size; i < rank; ++i) {
    inner_size_ *= inputs.at(0)->length(i);
  }
  return Status::OK();
}

CPULayerNorm::~CPULayerNorm() {
  if (gamma_.get()) {
    backend()->onReleaseBuffer(gamma_.get(), Device::STATIC);
  }
  if (beta_.get()) {
    backend()->onReleaseBuffer(beta_.get(), Device::STATIC);
  }
}

class CPULayerNormCreator : public CPUDevice::Creator {
 public:
  Operation* onCreate(const std::vector<Tensor*>& inputs,
                      const std::vector<Tensor*>& outputs, const tars::Op* op,
                      Device* backend) const override {
    return new CPULayerNorm(op, backend);
  }
};

REGISTER_CPU_OP_CREATOR(CPULayerNormCreator, OpType_LayerNorm);

}  // namespace tars

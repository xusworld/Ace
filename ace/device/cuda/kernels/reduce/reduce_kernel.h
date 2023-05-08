#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <functional>

namespace ace {
namespace device {
namespace cuda {

enum class ReduceOpType {
  ReduceUnknow = 0,
  ReduceMin,
  ReduceMax,
  ReduceSum,
  ReduceAvg,
  ReduceProd
};

namespace {

// Reduce Op
template <typename T, ReduceOpType op>
class ReduceOp {
 public:
  __device__ static T compute(T a, T b) { return T(-1); }
};

template <>
__device__ float ReduceOp<float, ReduceOpType::ReduceMin>::compute(float a,
                                                                   float b) {
  return ((a > b) ? b : a);
}

template <>
__device__ float ReduceOp<float, ReduceOpType::ReduceMax>::compute(float a,
                                                                   float b) {
  return ((a > b) ? a : b);
}

template <>
__device__ float ReduceOp<float, ReduceOpType::ReduceSum>::compute(float a,
                                                                   float b) {
  return a + b;
}

template <>
__device__ float ReduceOp<float, ReduceOpType::ReduceAvg>::compute(float a,
                                                                   float b) {
  return a + b;
}

template <>
__device__ float ReduceOp<float, ReduceOpType::ReduceProd>::compute(float a,
                                                                    float b) {
  return a * b;
}

}  // namespace

template <ReduceOpType op>
void ReduceCaller(const float* input, float* output, const size_t n);

void KernelExecTimecost(const std::string& title,
                        std::function<void(void)> func);

}  // namespace cuda
}  // namespace device
}  // namespace ace

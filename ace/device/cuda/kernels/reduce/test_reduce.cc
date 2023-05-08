#include <iostream>

#include "reduce_kernel.h"

template <class T>
T reduceCPU(T *data, int size) {
  T sum = data[0];
  T c = (T)0.0;

  for (int i = 1; i < size; i++) {
    T y = data[i] - c;
    T t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}

int main() {
  const size_t size = 1 << 25;

  // Allocate the host input vector A.
  float *host_inputs = (float *)malloc(size * sizeof(float));
  float *host_output = (float *)malloc(sizeof(float));

  // Init inputs.
  for (int i = 0; i < size; ++i) {
    host_inputs[i] = (rand() & 0xFF) / (float)RAND_MAX;
  }

  host_output[0] = 0.0f;

  // ace::device::cuda::ReduceCaller<ace::device::cuda::ReduceOpType::ReduceMin>(
  //     host_inputs, host_output, num_of_elements);

  // std::cout << "output: " << host_output[0] << std::endl;

  // ace::device::cuda::ReduceCaller<ace::device::cuda::ReduceOpType::ReduceMax>(
  //     host_inputs, host_output, num_of_elements);

  // std::cout << "output: " << host_output[0] << std::endl;

  ace::device::cuda::ReduceCaller<ace::device::cuda::ReduceOpType::ReduceSum>(
      host_inputs, host_output, size);

  std::cout << "CUDA ReduceSum output: " << host_output[0] << std::endl;

  std::cout << "CPU ReduceSum output: " << reduceCPU<float>(host_inputs, size)
            << std::endl;
  return 0;
}
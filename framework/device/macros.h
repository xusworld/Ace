#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

#include <iostream>
#include <list>
#include <map>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

const int CUDA_NUM_THREADS = 512;

#define CUDA_KERNEL_LE(i, n)                     \
  int i = blockIdx.x * blockDim.x + threadIdx.x; \
  if (i < n)

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

/// CUDA: number of blocks for threads.
inline int CUDA_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
inline int CUDA_GET_BLOCKS(const int N, const int base) {
  return (N + base - 1) / base;
}

#define CUDA_CHECK(condition)                                         \
  /* Code block avoids redefinition of cudaError_t error */           \
  do {                                                                \
    cudaError_t error = condition;                                    \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

#ifdef USE_CUBLAS
#include <cublas_v2.h>
#define CUBLAS_CHECK(condition)                                                \
  do {                                                                         \
    cublasStatus_t status = condition;                                         \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << cublas_get_errorstring(status); \
  } while (0)
const char* cublas_get_errorstring(cublasStatus_t error);
#endif  // USE_CUBLAS

#ifdef USE_CURAND
#include <curand.h>
#endif  // USE_CURAND

#ifdef USE_CUFFT
#include <cufft.h>
#endif  // USE_CUFFT

#ifdef USE_CUDNN
#include <cudnn.h>
#define CUDNN_VERSION_MIN(major, minor, patch) \
  (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDNN_CHECK(condition)                                               \
  do {                                                                       \
    cudnnStatus_t status = condition;                                        \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnn_get_errorstring(status); \
  } while (0)

const char* cudnn_get_errorstring(cudnnStatus_t status);
#endif  // USE_CUDNN
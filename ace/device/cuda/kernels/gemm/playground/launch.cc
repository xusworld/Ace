#include <cuda_runtime.h>

#include "kernels/sgemm.h"

#define CEIL_DIV(m, n) ((m) + (n)-1) / (n)

void launch_sgemm_v1(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  cudaDeviceSynchronize();
  dim3 blockDim(32, 32);
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  sgemm_v1<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();
}
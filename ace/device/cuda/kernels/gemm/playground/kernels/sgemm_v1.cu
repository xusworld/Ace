#include <stdio.h>
#include <stdlib.h>

#include "sgemm.h"

#define A(i, j) A[(i) + (j)*lda]
#define B(i, j) B[(i) + (j)*ldb]
#define C(i, j) C[(i) + (j)*ldc]

// naive version
__global__ void 
__launch_bounds__(1024)
sgemm_v1(int M, int N, int K, float alpha, float* A, float* B, float beta,
           float* C) {
  int lda = M;
  int ldb = K;
  int ldc = M;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  A = &A((bx << 5), 0);
  B = &B(0, (by << 5));
  C = &C((bx << 5), (by << 5));
  float tmp = 0.;

  for (int k_count = 0; k_count < K; k_count++) {
    tmp += A(tx, k_count) * B(k_count, ty);
  }

  C(tx, ty) = alpha * tmp + beta * C(tx, ty);
}



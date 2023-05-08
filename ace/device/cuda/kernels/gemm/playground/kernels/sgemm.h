#pragma once

#include <cuda_runtime.h>

__global__ void sgemm_v1(int M, int N, int K, float alpha, float* A, float* B,
                         float beta, float* C);
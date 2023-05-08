#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_string.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cstring>
#include <iostream>
#include <vector>

#include "launch.h"
#include "matrix.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    printf(
        "Please select a kernel (range 0 - 11, here 0 is for NVIDIA "
        "cuBLAS).\n");
    exit(-1);
  }

  // shapes is a vector of (M, N)
  std::vector<std::vector<int>> shapes;
  for (int i = 10; i <= 24; i++) {
    shapes.push_back(std::vector<int>(2, i << 8));
  }

  // parameters
  float alpha = 1.0f;
  float beta = 0.0f;

  const auto shape = shapes[0];
  std::cout << "Shape: " << shape[0] << " ," << shape[1] << std::endl;
  // Assume that matrix A, B and C share the same shape.
  const int32_t size_in_bytes = sizeof(float) * shape[0] * shape[1];

  // Allocate host memory for matrices.
  float *ha = (float *)malloc(size_in_bytes);
  float *hb = (float *)malloc(size_in_bytes);
  float *hc = (float *)malloc(size_in_bytes);
  float *hr = (float *)malloc(size_in_bytes);

  // Initialize host matrices.
  RandomizeMatrix(ha, shape);
  RandomizeMatrix(hb, shape);
  RandomizeMatrix(hc, shape);
  memcpy(hr, hc, size_in_bytes);

  float elapsed_time;

  // Create cuda event
  cublasHandle_t err;
  cublasCreate(&err);
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // Allocate device memory for matrices.
  float *da = NULL;
  float *db = NULL;
  float *dc = NULL;
  float *dr = NULL;
  checkCudaErrors(cudaMalloc((void **)&da, size_in_bytes));
  checkCudaErrors(cudaMalloc((void **)&db, size_in_bytes));
  checkCudaErrors(cudaMalloc((void **)&dc, size_in_bytes));
  checkCudaErrors(cudaMalloc((void **)&dr, size_in_bytes));
  // H2D
  checkCudaErrors(cudaMemcpy(da, ha, size_in_bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(db, hb, size_in_bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dc, hc, size_in_bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dr, hr, size_in_bytes, cudaMemcpyHostToDevice));

  const int times = 1;
  for (int i = 0; i < times; ++i) {
    const int M = shape[0];
    const int N = M;
    const int K = M;

    cublasSgemm(err, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, da, M, db, K,
                &beta, dc, M);
    cudaDeviceSynchronize();

    launch_sgemm_v1(M, N, K, &alpha, da, M, db, K, &beta, dc, M);

    // DTH
    checkCudaErrors(cudaMemcpy(hc, dc, size_in_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hr, dr, size_in_bytes, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
  }
  // Sync
  cudaDeviceSynchronize();

  std::cout << "Free allocated host memory." << std::endl;
  // Deallocated host memory.
  free(ha);
  free(hb);
  free(hc);
  free(hr);

  std::cout << "Free allocated device memory." << std::endl;
  // Deallocated device memory.
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cudaFree(dr);

  // Sync
  cudaDeviceSynchronize();
  return 0;
}

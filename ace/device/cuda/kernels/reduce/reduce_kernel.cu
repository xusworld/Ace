#include "../utils/helper_cuda.h"
#include "reduce_kernel.h"
#include <iostream>

namespace ace {
namespace device {
namespace cuda {

template <ReduceOpType op, int32_t BlockSize>
__device__ void ReduceSharedMemory(float *shm, float *result) {
  if (BlockSize >= 1024) {
    if (threadIdx.x < 512) {
      shm[threadIdx.x] = ReduceOp<float, op>::compute(shm[threadIdx.x],
                                                      shm[threadIdx.x + 512]);
    }
    __syncthreads();
  }

  if (BlockSize >= 512) {
    if (threadIdx.x < 256) {
      shm[threadIdx.x] = ReduceOp<float, op>::compute(shm[threadIdx.x],
                                                      shm[threadIdx.x + 256]);
    }
    __syncthreads();
  }

  if (BlockSize >= 256) {
    if (threadIdx.x < 128) {
      shm[threadIdx.x] = ReduceOp<float, op>::compute(shm[threadIdx.x],
                                                      shm[threadIdx.x + 128]);
    }
    __syncthreads();
  }

  if (BlockSize >= 128) {
    if (threadIdx.x < 64) {
      shm[threadIdx.x] =
          ReduceOp<float, op>::compute(shm[threadIdx.x], shm[threadIdx.x + 64]);
    }
    __syncthreads();
  }

  // mark the final warp
  if (threadIdx.x < 32) {
    volatile float *vshm = shm;
    if (blockDim.x >= 64) {
      vshm[threadIdx.x] = ReduceOp<float, op>::compute(vshm[threadIdx.x],
                                                       vshm[threadIdx.x + 32]);
    }
    vshm[threadIdx.x] =
        ReduceOp<float, op>::compute(vshm[threadIdx.x], vshm[threadIdx.x + 16]);
    vshm[threadIdx.x] =
        ReduceOp<float, op>::compute(vshm[threadIdx.x], vshm[threadIdx.x + 8]);
    vshm[threadIdx.x] =
        ReduceOp<float, op>::compute(vshm[threadIdx.x], vshm[threadIdx.x + 4]);
    vshm[threadIdx.x] =
        ReduceOp<float, op>::compute(vshm[threadIdx.x], vshm[threadIdx.x + 2]);
    vshm[threadIdx.x] =
        ReduceOp<float, op>::compute(vshm[threadIdx.x], vshm[threadIdx.x + 1]);

    if (threadIdx.x == 0) {
      *result = vshm[0];
    }
  }
}

__device__ int32_t done_block_count = 0;

template <ReduceOpType op, int32_t block_threads_num>
__global__ void ParallelReduceImpl(const float *input, float *part_sum,
                                   float *output, const size_t n) {
  // unique thread id
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  // total launched threads for kernel
  int32_t total_threads_num = gridDim.x * blockDim.x;

  float res = 0.0f;
  for (int32_t i = tid; i < n; i += total_threads_num) {
    res = ReduceOp<float, op>::compute(res, input[i]);
  }

  // store sum to shared memory
  extern __shared__ float shm[];
  shm[threadIdx.x] = res;
  __syncthreads();

  // reduce shared memory to part_sum
  ReduceSharedMemory<op, block_threads_num>(shm, part_sum + blockIdx.x);

  // make sure when a block get is_last_block is true,
  // all the other part_sums is ready
  __threadfence();

  // check if this block is the last
  __shared__ bool is_last_block;
  if (threadIdx.x == 0) {
    is_last_block = atomicAdd(&done_block_count, 1) == gridDim.x - 1;
  }
  __syncthreads();

  // reduce part_sum to output
  if (is_last_block) {
    res = 0.0f;
    for (int32_t i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
      res = ReduceOp<float, op>::compute(res, part_sum[i]);
    }
    shm[threadIdx.x] = res;
    __syncthreads();

    ReduceSharedMemory<op, block_threads_num>(shm, output);
    done_block_count = 0;
  }
}

template <ReduceOpType op>
void ReduceCaller(const float *host_inputs, float *host_output,
                  const size_t n) {
  const int32_t grid_size = 1024;
  const int32_t block_size = 1024;

  // 1024 x 4 bytes = 4M shared memory for each thread block
  size_t shm_size = block_size * sizeof(float);

  // total size
  const size_t size_in_bytes = n * sizeof(float);

  // raw inputs
  float *device_inputs = NULL;
  checkCudaErrors(cudaMalloc((void **)&device_inputs, size_in_bytes));
  // partial outputs
  float *device_partial_sum = NULL;
  checkCudaErrors(
      cudaMalloc((void **)&device_partial_sum, grid_size * sizeof(float)));
  // final outputs
  float *device_output = NULL;
  checkCudaErrors(cudaMalloc((void **)&device_output, sizeof(float)));

  // Initialize inputs
  cudaMemcpy(device_inputs, host_inputs, size_in_bytes, cudaMemcpyHostToDevice);
  cudaMemset(device_partial_sum, 0.0f, grid_size * sizeof(float));
  cudaMemset(device_output, 0.0f, sizeof(float));

  KernelExecTimecost("ReduceBySinglePass", [&]() {
    ParallelReduceImpl<op, block_size><<<grid_size, block_size, shm_size>>>(
        device_inputs, device_partial_sum, device_output, n);
  });

  cudaMemcpy(host_output, device_output, sizeof(float), cudaMemcpyDeviceToHost);

  // Deallocate device memory.
  cudaFree(device_inputs);
  cudaFree(device_partial_sum);
  cudaFree(device_output);
}

template void ReduceCaller<ReduceOpType::ReduceMin>(const float *host_inputs,
                                                    float *host_output,
                                                    const size_t n);

template void ReduceCaller<ReduceOpType::ReduceMax>(const float *host_inputs,
                                                    float *host_output,
                                                    const size_t n);

template void ReduceCaller<ReduceOpType::ReduceSum>(const float *host_inputs,
                                                    float *host_output,
                                                    const size_t n);

template void ReduceCaller<ReduceOpType::ReduceAvg>(const float *host_inputs,
                                                    float *host_output,
                                                    const size_t n);

template void ReduceCaller<ReduceOpType::ReduceProd>(const float *host_inputs,
                                                     float *host_output,
                                                     const size_t n);

void KernelExecTimecost(const std::string &title,
                        std::function<void(void)> func) {
  cudaEvent_t start, stop;
  func();
  func();
  func();
  func();
  func();

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  func();

  // record stop event on the default stream
  cudaEventRecord(stop);
  // wait until the stop event completes
  cudaEventSynchronize(stop);
  // calculate the elapsed time between two events
  float time;
  cudaEventElapsedTime(&time, start, stop);

  std::cout << title << " , timecost is " << time << " ms" << std::endl;

  // clean up the two events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

} // namespace cuda
} // namespace device
} // namespace ace
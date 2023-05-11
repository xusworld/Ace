#include "cuda_runtime.h"
#include "framework/device/status.h"
#include "macros.h"

namespace ace {
namespace device {

Status CudaDevice::AllocMemory(void** ptr, const int32_t size) {
  CUDA_CHECK(cudaMallocHost(ptr, size));
  return Status::OK();
}

Status CudaDevice::FreeMemory(void* ptr) {
  if (ptr != nullptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
  }
  return Status::OK();
}

Status CudaDevice::ResetMemory(void* ptr, const int32_t val,
                               const int32_t size) {
  memset(ptr, val, size);
  return Status::OK();
}

Status CudaDevice::SyncMemcpy(void* dst, size_t dst_offset, int dst_id,
                              const void* src, size_t src_offset, int src_id,
                              size_t count, MemcopyKind) {
  CUDA_CHECK(cudaMemcpy((char*)dst + dst_offset, (char*)src + src_offset, count,
                        cudaMemcpyHostToHost));
  CUDA_CHECK(cudaStreamSynchronize(0));
  //LOG(INFO) << "NVH, sync, H2H, size: " << count << ", src_offset: " \
          << src_offset << ", data:" << ((const float*)((char*)src + src_offset))[0];
  return Status::OK();
}

Status CudaDevice::AsyncMemcpy(void* dst, size_t dst_offset, int dst_id,
                               const void* src, size_t src_offset, int src_id,
                               size_t count, MemcopyKind) {
  CUDA_CHECK(cudaMemcpy((char*)dst + dst_offset, (char*)src + src_offset, count,
                        cudaMemcpyHostToHost));
  // LOG(INFO) << "NVH, sync, H2H, size: " << count;
  return Status::OK();
}

Status CudaDevice::CreateEvent(event_t* event, bool flag) {
  if (flag) {
    CUDA_CHECK(cudaEventCreateWithFlags(event, cudaEventDefault));
  } else {
    CUDA_CHECK(cudaEventCreateWithFlags(event, cudaEventDisableTiming));
  }
  return Status::OK();
}

Status CudaDevice::DestroyEvent(event_t event) {
  CUDA_CHECK(cudaEventDestroy(event));
  return Status::OK();
}

Status CudaDevice::RecordEvent(event_t event, stream_t stream) {
  CUDA_CHECK(cudaEventRecord(event, stream));
  return Status::OK();
}

Status CudaDevice::QueryEvent(event_t event) {
  CUDA_CHECK(cudaEventQuery(event));
  return Status::OK();
}

Status CudaDevice::SyncEvent(event_t event) {
  CUDA_CHECK(cudaEventSynchronize(event));
  return Status::OK();
}

Status CudaDevice::CreateStream(stream_t* stream) {
  CUDA_CHECK(cudaStreamCreate(stream));
  return Status::OK();
}

Status CudaDevice::CreateStreamWithFlag(stream_t* stream, unsigned int flag) {
  CUDA_CHECK(cudaStreamCreateWithFlags(stream, flag));
  return Status::OK();
}

Status CudaDevice::CreateStreamWithPriority(stream_t* stream, unsigned int flag,
                                            int priority) {
  CUDA_CHECK(cudaStreamCreateWithPriority(stream, flag, priority));
  return Status::OK();
}

Status CudaDevice::DestroyStream(stream_t stream) {
  CUDA_CHECK(cudaStreamDestroy(stream));
  return Status::OK();
}

Status CudaDevice::SyncStream(event_t event, stream_t stream) {
  CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
  return Status::OK();
}

Status CudaDevice::SyncStream(stream_t stream) {
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return Status::OK();
}
}  // namespace device
}  // namespace ace
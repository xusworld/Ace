#include "cpu_runtime.h"

namespace ace {
namespace device {

const int MALLOC_ALIGN = 64;

static inline void* fast_malloc(size_t size) {
  size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
  char* p = static_cast<char*>(malloc(offset + size));

  if (!p) {
    return nullptr;
  }

  void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) &
                                    (~(MALLOC_ALIGN - 1)));
  static_cast<void**>(r)[-1] = p;
  memset(r, 0, size);
  return r;
}

static inline void fast_free(void* ptr) {
  if (ptr) {
    free(static_cast<void**>(ptr)[-1]);
  }
}

Status CpuRuntime::AllocMemory(void** ptr, const int32_t size) {
  *ptr = (void*)fast_malloc(size);
  return Status::OK();
}

Status CpuRuntime::FreeMemory(void* ptr) {
  if (ptr != nullptr) {
    fast_free(ptr);
  }
  return Status::OK();
}

Status CpuRuntime::ResetMemory(void* ptr, const int32_t val,
                               const int32_t size) {
  memset(ptr, val, size);
  return Status::OK();
}

Status CpuRuntime::SyncMemcpy(void* dst, size_t dst_offset, int dst_id,
                              const void* src, size_t src_offset, int src_id,
                              size_t count, MemcopyKind) {
  memcpy((char*)dst + dst_offset, (char*)src + src_offset, count);
  return Status::OK();
}

Status CpuRuntime::AsyncMemcpy(void* dst, size_t dst_offset, int dst_id,
                               const void* src, size_t src_offset, int src_id,
                               size_t count, MemcopyKind) {
  memcpy((char*)dst + dst_offset, (char*)src + src_offset, count);
  return Status::OK();
}

Status CpuRuntime::CreateEvent(event_t* event, bool flag) {
  return Status::OK();
}

Status CpuRuntime::DestroyEvent(event_t event) { return Status::OK(); }

Status CpuRuntime::RecordEvent(event_t event, stream_t stream) {
  return Status::OK();
}

Status CpuRuntime::QueryEvent(event_t event) { return Status::OK(); }

Status CpuRuntime::SyncEvent(event_t event) { return Status::OK(); }

Status CpuRuntime::CreateStream(stream_t* stream) { return Status::OK(); }

Status CpuRuntime::CreateStreamWithFlag(stream_t* stream, unsigned int flag) {
  return Status::OK();
}

Status CpuRuntime::CreateStreamWithPriority(stream_t* stream, unsigned int flag,
                                            int priority) {
  return Status::OK();
}

Status CpuRuntime::DestroyStream(stream_t stream) { return Status::OK(); }

Status CpuRuntime::SyncStream(event_t event, stream_t stream) {
  return Status::OK();
}

Status CpuRuntime::SyncStream(stream_t stream) { return Status::OK(); }

}  // namespace device
}  // namespace ace
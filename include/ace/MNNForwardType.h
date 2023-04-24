//
//  types.h
//  MNN
//
//  Created by MNN on 2019/01/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNForwardType_h
#define MNNForwardType_h
#include <stddef.h>
#include <stdint.h>

typedef enum {
  // choose one tuning mode Only
  MNN_GPU_TUNING_NONE = 1 << 0,   /* Forbidden tuning, performance not good */
  MNN_GPU_TUNING_HEAVY = 1 << 1,  /* heavily tuning, usually not suggested */
  MNN_GPU_TUNING_WIDE = 1 << 2,   /* widely tuning, performance good. Default */
  MNN_GPU_TUNING_NORMAL = 1 << 3, /* normal tuning, performance may be ok */
  MNN_GPU_TUNING_FAST = 1 << 4,   /* fast tuning, performance may not good */

  // choose one opencl memory mode Only
  /* User can try OpenCL_MEMORY_BUFFER and OpenCL_MEMORY_IMAGE both,
   then choose the better one according to performance*/
  MNN_GPU_MEMORY_BUFFER = 1 << 6, /* User assign mode */
  MNN_GPU_MEMORY_IMAGE = 1 << 7,  /* User assign mode */
} MNNGpuMode;

#ifdef __cplusplus
namespace ace {};  // namespace ace
#endif
#endif /* MNNForwardType_h */

#ifndef RASTER_CU_H
#define RASTER_CU_H
#include "CommonPlugin.hpp"
#include "core/TensorUtils.hpp"
#include <cuda_runtime_api.h>

namespace ace {
cudaError_t RasterBlit(nvinfer1::DataType dataType, uint8_t *dest,
                       const uint8_t *src,
                       const Tensor::InsideDescribe::Region &reg, int bytes,
                       cudaStream_t stream);
}

#endif
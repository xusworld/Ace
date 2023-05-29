//
//  ConvSingleInputoperation.h
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvSingleInputExecution_hpp
#define ConvSingleInputExecution_hpp

#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {

struct KernelInfo {
  int groups = 0;
  int kernelN = 0;
  int kernelC = 0;
  int kernelX = 0;
  int kernelY = 0;
  int strideX = 0;
  int strideY = 0;
  int dilateX = 0;
  int dilateY = 0;
  int activationType = 0;
};  //

}  // namespace cuda
}  // namespace tars

#endif /* ConvSingleInputExecution */

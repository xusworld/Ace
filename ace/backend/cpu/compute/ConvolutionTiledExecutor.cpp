//
//  ConvolutionTiledExecutor.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <ace/AutoTime.hpp>

#include "CommonOptFunction.h"
#include "ConvOpt.h"
#include "ConvolutionTiledExecutor.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/BufferAllocator.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/MemoryFormater.h"
#include "core/TensorUtils.hpp"
#include "math/Vec.hpp"

using Vec4 = ace::Math::Vec<float, 4>;
namespace ace {

void ConvolutionTiledExecutor::initWeight(const float* source, float* cache,
                                          int depth, int outputCount,
                                          int kernelSize,
                                          const CoreFunctions* function) {
  // Swap k, ic
  int dims[4] = {depth, kernelSize, kernelSize, depth};
  for (int o = 0; o < outputCount; ++o) {
    auto dO = cache + o * depth * kernelSize;
    auto sO = source + o * depth * kernelSize;
    MNNTranspose32Bit((int32_t*)dO, (const int32_t*)sO, &dims[0]);
  }
  if (function->bytes < 4) {
    // Lowp
    function->MNNFp32ToLowp((float*)cache, (int16_t*)cache,
                            outputCount * kernelSize * depth);
  }
}

ConvolutionTiledExecutor::ConvolutionTiledExecutor(Backend* b,
                                                   const float* bias,
                                                   size_t biasSize)
    : ace::Execution(b) {
  mResource.reset(new CPUConvolution::Resource);
  mResource->backend = b;
  mValid = mResource->copyBiasAlign(bias, biasSize);
  if (!mValid) {
    return;
  }
}

ConvolutionTiledExecutor::ConvolutionTiledExecutor(
    std::shared_ptr<CPUConvolution::Resource> res, Backend* b)
    : mResource(res), Execution(b) {}

ConvolutionTiledExecutor::~ConvolutionTiledExecutor() {
  // Do nothing
}
bool ConvolutionTiledExecutor::onClone(Backend* bn, const Op* op,
                                       Execution** dst) {
  if (!mValid) {
    return false;
  }
  if (nullptr == dst) {
    return true;
  }
  *dst = new ConvolutionTiledExecutor(mResource, bn);
  return true;
}

ErrorCode ConvolutionTiledImpl::onResize(const std::vector<Tensor*>& inputs,
                                         const std::vector<Tensor*>& outputs) {
  return NO_ERROR;
}

ErrorCode ConvolutionTiledImpl::onExecute(const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs) {
  MNN_CONCURRENCY_BEGIN(tId, mFunction.first) { mFunction.second((int)tId); }
  MNN_CONCURRENCY_END();
  return NO_ERROR;
}

}  // namespace ace

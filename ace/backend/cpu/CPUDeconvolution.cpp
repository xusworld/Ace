//
//  CPUDeconvolution.cpp
//  MNN
//
//  Created by MNN on 2018/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUBackend.hpp"
#include "CPUDeconvolution.hpp"
#include "compute/CommonOptFunction.h"
#include "compute/ConvOpt.h"
#include "compute/DeconvolutionWithStride.hpp"
#include "core/AutoStorage.h"
#include "core/BufferAllocator.hpp"
#include "core/Concurrency.h"
#include "core/ConvolutionCommon.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "math/Matrix.hpp"
// #define MNN_OPEN_TIME_TRACE
#include <ace/AutoTime.hpp>

namespace ace {
CPUDeconvolutionBasic::CPUDeconvolutionBasic(const Tensor* input,
                                             const Op* convOp, Backend* b)
    : CPUConvolution(convOp->main_as_Convolution2D()->common(), b) {
  mSrcCount = input->channel();
  mPostParameters = getPostParameters();
}

ErrorCode CPUDeconvolutionBasic::onResize(const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs) {
  auto input = inputs[0];
  auto output = outputs[0];
  auto pad = ConvolutionCommon::convolutionTransposePad(input, output, mCommon);
  mPadY = pad.second;
  mPadX = pad.first;
  return NO_ERROR;
}

CPUDeconvolutionCommon::CPUDeconvolutionCommon(const Tensor* input,
                                               const Op* convOp, Backend* b)
    : CPUDeconvolutionBasic(input, convOp, b) {
  auto conv2D = convOp->main_as_Convolution2D();
  int outputCount = mCommon->outputCount();
  auto core = static_cast<CPUBackend*>(b)->functions();
  mBias.reset(Tensor::createDevice<float>(
      std::vector<int>{UP_DIV(outputCount, core->pack) * core->pack}));
  bool success = b->onAcquireBuffer(mBias.get(), Backend::STATIC);
  if (!success) {
    mValid = false;
    return;
  }
  ::memset(mBias->host<float>(), 0, mBias->length(0) * core->bytes);
  if (core->bytes == 4) {
    ::memcpy(mBias->host<float>(), conv2D->bias()->data(),
             conv2D->bias()->size() * sizeof(float));
  } else {
    core->MNNFp32ToLowp(conv2D->bias()->data(), mBias->host<int16_t>(),
                        conv2D->bias()->size());
  }
}

CPUDeconvolutionCommon::~CPUDeconvolutionCommon() {
  backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
}

static void _transformWeight(const uint8_t* tempWeight, uint8_t* dest,
                             int outputCount, int srcCount, int fh, int fw,
                             uint8_t* cache, const CoreFunctions* core) {
  auto outputC4 = UP_DIV(outputCount, core->pack);
  // c, n, h, w-> c, n/4 * 4, h, w
  for (int c = 0; c < srcCount; ++c) {
    auto dst = cache + c * outputC4 * fw * fh * core->pack * core->bytes;
    auto src = tempWeight + c * outputCount * fw * fh * core->bytes;
    core->MNNPackCUnit((float*)dst, (const float*)src, fw * fh, outputCount);
  }
  // printf("%d - %d - %d - %d\n", outputCount, srcCount, fh, fw);
  core->MNNPackForMatMul_B((float*)dest, (const float*)cache,
                           outputC4 * fw * fh * core->pack, srcCount, false);
}

CPUDeconvolution::CPUDeconvolution(const Tensor* input, const Op* convOp,
                                   Backend* backend)
    : ace::CPUDeconvolutionCommon(input, convOp, backend) {
  auto layer = convOp->main_as_Convolution2D()->common();
  auto core = static_cast<CPUBackend*>(backend)->functions();

  const float* tempWeight = nullptr;
  int tempWeightSize = 0;
  std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
  ConvolutionCommon::getConvParameters(&quanCommon,
                                       convOp->main_as_Convolution2D(),
                                       &tempWeight, &tempWeightSize);

  int fw = layer->kernelX();
  int fh = layer->kernelY();
  int srcCount = mSrcCount;
  int eP, lP, hP;
  core->MNNGetMatMulPackMode(&eP, &lP, &hP);
  auto outputAlign =
      UP_DIV(layer->outputCount(), core->pack) * core->pack * fw * fh;
  mWeight.reset(Tensor::createDevice<float>(std::vector<int>{
      UP_DIV(outputAlign, hP), UP_DIV(srcCount, lP) * lP, hP}));
  std::shared_ptr<Tensor> cache(
      Tensor::createDevice<float>({outputAlign * srcCount}));
  bool success = backend->onAcquireBuffer(mWeight.get(), Backend::STATIC) &&
                 backend->onAcquireBuffer(cache.get(), Backend::STATIC);
  if (!success) {
    mValid = false;
    return;
  }
  auto dest = mWeight->host<uint8_t>();
  int outputCount = layer->outputCount();
  AutoStorage<uint8_t> lowpWeight;
  if (core->bytes < 4) {
    lowpWeight.reset(outputCount * srcCount * fh * fw * core->bytes);
    if (lowpWeight.get() == nullptr) {
      mValid = false;
      return;
    }
    core->MNNFp32ToLowp(tempWeight, (int16_t*)lowpWeight.get(),
                        outputCount * srcCount * fh * fw);
    tempWeight = (float*)lowpWeight.get();
  }
  _transformWeight((uint8_t*)tempWeight, dest, outputCount, srcCount, fh, fw,
                   cache->host<uint8_t>(), core);
  backend->onReleaseBuffer(cache.get(), Backend::STATIC);
  mOrigin.reset(new CPUDeconvolutionOrigin(input, convOp, backend));
}

CPUDeconvolution::~CPUDeconvolution() {
  backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
}

ErrorCode CPUDeconvolutionOrigin::onResize(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
  CPUDeconvolutionBasic::onResize(inputs, outputs);
  auto core = static_cast<CPUBackend*>(backend())->functions();
  auto input = inputs[0];
  auto output = outputs[0];
  auto oc = output->channel();
  if (UP_DIV(oc, core->pack) * core->pack != inputs[2]->length(0)) {
    return INPUT_DATA_ERROR;
  }

  auto ocC4 = UP_DIV(output->channel(), core->pack);
  auto icC4 = UP_DIV(input->channel(), core->pack);
  auto kw = mCommon->kernelX();
  auto kh = mCommon->kernelY();
  auto dilateX = mCommon->dilateX();
  auto dilateY = mCommon->dilateY();
  auto strideX = mCommon->strideX();
  auto strideY = mCommon->strideY();
  auto padX = mPadX;
  auto padY = mPadY;
  auto width = input->width();
  auto height = input->height();
  auto src_height = output->height();
  auto src_width = output->width();

  auto kernelCount = ocC4 * mCommon->kernelX() * mCommon->kernelY();
  mPreFunctions.clear();
  mPostFunctions.clear();
  auto plane = width * height;
  const int maxDepth = 5;
  AutoRelease<Tensor> tempColTotalBuffer(
      Tensor::createDevice<float>({kernelCount, plane, core->pack}));
  auto res =
      backend()->onAcquireBuffer(tempColTotalBuffer.get(), Backend::DYNAMIC);
  if (!res) {
    return OUT_OF_MEMORY;
  }
  auto colBufferPtr = tempColTotalBuffer->host<float>();
  auto biasPtr = inputs[2]->host<float>();
  auto inputPtr = input->host<float>();
  AutoRelease<Tensor> tempInputBuffer(
      Tensor::create<float>({icC4, plane, core->pack}, inputPtr));
  AutoRelease<Tensor> tempInput(
      Tensor::createDevice<float>({icC4, plane, core->pack}));
  auto threadNumber = ((CPUBackend*)backend())->threadNumber();
  if (input->batch() != 1) {
    res = backend()->onAcquireBuffer(tempInput.get(), Backend::DYNAMIC);
    if (!res) {
      return OUT_OF_MEMORY;
    }
    auto newInputPtr = tempInput->host<uint8_t>();
    // Copy Batch
    mPreFunctions.emplace_back(std::make_pair(
        [newInputPtr, icC4, plane, threadNumber, core](const float* srcBatch,
                                                       int tId) {
          for (int c = tId; c < icC4; c += threadNumber) {
            auto srcDepth =
                ((uint8_t*)srcBatch) + c * plane * core->pack * core->bytes;
            auto dstDepth = newInputPtr + c * plane * core->pack * core->bytes;
            ::memcpy(dstDepth, srcDepth, plane * core->pack * core->bytes);
          }
        },
        threadNumber));
  } else {
    tempInput->buffer().host = (uint8_t*)inputPtr;
  }
  mMatMul.reset(new StrassenMatrixComputor(backend(), true, maxDepth));
  mMatMul->onEncode({tempInput.get(), inputs[1]}, {tempColTotalBuffer.get()});
  mPostFunctions.emplace_back(std::make_pair(
      [colBufferPtr, ocC4, width, height, kh, kw, padY, padX, dilateY, dilateX,
       strideY, strideX, threadNumber, src_width, src_height, plane, biasPtr,
       this, core](float* outputPtr, int tId) {
        auto unitBytes = core->pack * core->bytes;
        for (int z = (tId); z < ocC4; z += threadNumber) {
          auto dstZ =
              (uint8_t*)outputPtr + z * src_height * src_width * unitBytes;
          auto srcZ = (uint8_t*)colBufferPtr + kw * kh * plane * z * unitBytes;
          auto dstB = dstZ;
          ::memset(dstB, 0, src_width * src_height * unitBytes);
          auto srcB = srcZ;
          for (int oy = 0; oy < height; ++oy) {
            for (int ox = 0; ox < width; ++ox) {
              int srcStartX = ox * strideX - padX;
              int srcStartY = oy * strideY - padY;

              int sfy = ALIMAX(0, (UP_DIV(-srcStartY, dilateY)));
              int efy = ALIMIN(kh, UP_DIV(src_height - srcStartY, dilateY));

              int sfx = ALIMAX(0, (UP_DIV(-srcStartX, dilateX)));
              int efx = ALIMIN(kw, UP_DIV(src_width - srcStartX, dilateX));

              auto dstStart = dstB + srcStartX * unitBytes +
                              srcStartY * src_width * unitBytes;
              auto srcStart = srcB + unitBytes * (ox + oy * width);
              if (sfy >= efy || sfx >= efx) {
                continue;
              }

              for (int fy = sfy; fy < efy; ++fy) {
                auto dstY = dstStart + fy * unitBytes * dilateY * src_width;
                auto srcY = srcStart + fy * kw * plane * unitBytes;
                core->MNNAddC4WithStride(
                    (const float*)(srcY + sfx * plane * unitBytes),
                    (float*)(dstY + sfx * dilateX * unitBytes),
                    plane * core->pack, dilateX * core->pack, efx - sfx);
              }
            }
          }
          core->MNNAxByClampBroadcastUnit(
              (float*)dstZ, (float*)dstZ,
              (const float*)((uint8_t*)biasPtr + unitBytes * z),
              src_height * src_width, 0, 0, 1, mPostParameters.data());
        }
      },
      threadNumber));
  if (tempInput->host<float>() != inputPtr) {
    backend()->onReleaseBuffer(tempInput.get(), Backend::DYNAMIC);
  }
  backend()->onReleaseBuffer(tempColTotalBuffer.get(), Backend::DYNAMIC);
  return NO_ERROR;
}

ErrorCode CPUDeconvolutionOrigin::onExecute(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
  auto batch = inputs[0]->batch();
  auto core = static_cast<CPUBackend*>(backend())->functions();
  auto input = inputs[0];
  auto output = outputs[0];
  auto oc = output->channel();
  auto ocC4 = UP_DIV(output->channel(), core->pack);
  auto icC4 = UP_DIV(input->channel(), core->pack);
  auto width = output->width();
  auto height = output->height();
  auto src_height = input->height();
  auto src_width = input->width();
  for (int i = 0; i < batch; ++i) {
    auto inputPtr = inputs[0]->host<uint8_t>() + i * src_width * src_height *
                                                     icC4 * core->pack *
                                                     core->bytes;
    auto outputPtr = outputs[0]->host<uint8_t>() +
                     i * width * height * ocC4 * core->pack * core->bytes;
    for (auto& unit : mPreFunctions) {
      MNN_CONCURRENCY_BEGIN(tId, unit.second) {
        unit.first((float*)inputPtr, (int)tId);
      }
      MNN_CONCURRENCY_END();
    }
    mMatMul->onExecute();
    for (auto& unit : mPostFunctions) {
      MNN_CONCURRENCY_BEGIN(tId, unit.second) {
        unit.first((float*)outputPtr, (int)tId);
      }
      MNN_CONCURRENCY_END();
    }
  }
  return NO_ERROR;
}
class CPUDeconvolutionCreator : public CPUBackend::Creator {
 public:
  virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs,
                              const ace::Op* op, Backend* backend) const {
    auto convOp = op->main_as_Convolution2D();
    auto common = convOp->common();
    if (backend->type() == DeviceType::X86) {
      if (common->strideY() > 1 || common->strideX() > 1) {
        if (common->dilateX() == 1 && common->dilateY() == 1) {
          if (common->kernelX() / common->strideX() > 2 ||
              common->kernelY() / common->strideY() > 2) {
            return new DeconvolutionWithStride(inputs[0], op, backend);
          }
        }
      }
    }
    return new CPUDeconvolution(inputs[0], op, backend);
  }
};

REGISTER_CPU_OP_CREATOR(CPUDeconvolutionCreator, OpType_Deconvolution);
}  // namespace ace

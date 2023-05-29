//
//  ConvInt8Cutlassoperation.h
//  MNN
//
//  Created by MNN on 2023/01/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef ConvInt8CutlassExecution_hpp
#define ConvInt8CutlassExecution_hpp

#include "../MNNCUDADefine.hpp"
#include "../MNNCUDAFunction.cuh"
#include "CutlassGemmInt8Param.hpp"
#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {

typedef enum {
  GEMM_SIZE_NORMAL = 0,
  GEMM_SIZE_LITTLE = 1,
  GEMM_SIZE_LARGE = 2
} GemmSizeLevel;

class ConvInt8CutlassExecution : public Operation {
 public:
  struct Resource {
    Resource(Device* bn, const tars::Op* op);
    ~Resource();
    void* mWeightInt8Ptr;
    void* mBiasInt32Ptr;
    void* mScaleFloatPtr;
    std::shared_ptr<Tensor> mWeightInt8Tensor;
    std::shared_ptr<Tensor> mBiasInt32Tensor;
    std::shared_ptr<Tensor> mScaleFloatTensor;

    int32_t* mBiasInt32Vec;
    float* mScaleFloatVec;
    Device* mBackend = nullptr;

    // relu or relu6
    int mActivationType;
    int mActBits;

    int32_t mInputZeroPoint;
    int32_t mOutputZeroPoint;
    int8_t mClampMin;
    int8_t mClampMax;
    float mInputScale;
    float mOutputScale;
    int mOutputChannelPack;
    std::vector<int> mInt8WeightKernelSum;

    std::once_flag flag;
    void updateInputOutputScale(std::vector<float> inputQuantInfo,
                                std::vector<float> outputQuantInfo);
  };
  ConvInt8CutlassExecution(Device* backend, const tars::Op* op,
                           std::shared_ptr<Resource> res);
  virtual ~ConvInt8CutlassExecution();
  virtual Status onResize(const std::vector<Tensor*>& inputs,
                          const std::vector<Tensor*>& outputs) override;
  virtual Status onExecute(const std::vector<Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs) override;
  virtual bool onClone(Device* bn, const Op* op, Operation** dst) override;

  Status callCutlassGemmInt8TensorCore(const std::vector<Tensor*>& inputs,
                                       const std::vector<Tensor*>& outputs);
  Status callCutlassGemmInt8TensorCore16832(
      const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);

 private:
  std::shared_ptr<Resource> mResource;

  const Op* mOp = nullptr;
  CutlassGemmInfo mGemmInfo;

  ConvolutionCommon::Im2ColParameter mIm2ColParamter;
  std::pair<void*, int> mGpuIm2ColParam;

  void* mIm2ColBuffer;

  bool mIsConv1x1S1D1P0 = false;
  bool mNeedIm2Col = true;
  std::pair<void*, int> mGpuKernelParam;
  bool mIsBlock = false;
  int mBlockNum = 1;

  GemmInt8Tensor_Clamp_AlignTensor_Little mGemmInt8ClampLittle;
  GemmInt8Tensor_Clamp_AlignTensor_Normal mGemmInt8ClampNormal;
  GemmInt8Tensor_Clamp_AlignTensor_Large mGemmInt8ClampLarge;

  GemmInt8Tensor_Clamp_AlignTensor_Normal_Sm80 mGemmInt8ClampNormalSm80;

  GemmSizeLevel mGemmShapeSizeLevel = GEMM_SIZE_NORMAL;
  int mGpuComputeCap = 75;
  int mActivationType = 0;
  std::shared_ptr<Tensor> workspaceTensor;
  void* mWorkspace;
};

}  // namespace cuda
}  // namespace tars

#endif /* ConvInt8CutlassExecution */
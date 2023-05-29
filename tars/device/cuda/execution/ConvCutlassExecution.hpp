//
//  ConvCutlassoperation.h
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef ConvCutlassExecution_hpp
#define ConvCutlassExecution_hpp

#include "CutlassGemmParam.hpp"
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"
#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {

class ConvCutlassExecution : public Operation {
 public:
  struct Resource {
    Resource(Device* bn, const tars::Op* op);
    ~Resource();
    void* mFilter;
    void* mBias;
    std::shared_ptr<Tensor> weightTensor;
    std::shared_ptr<Tensor> biasTensor;
    Device* mBackend = nullptr;
  };
  ConvCutlassExecution(Device* backend, const tars::Op* op,
                       std::shared_ptr<Resource> res);
  virtual ~ConvCutlassExecution();
  virtual Status onResize(const std::vector<Tensor*>& inputs,
                          const std::vector<Tensor*>& outputs) override;
  virtual Status onExecute(const std::vector<Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs) override;
  virtual bool onClone(Device* bn, const Op* op, Operation** dst) override;

  Status callCutlassGemmCudaCoreFloat16(const std::vector<Tensor*>& inputs,
                                        const std::vector<Tensor*>& outputs);
  Status callCutlassGemmCudaCoreFloat32(const std::vector<Tensor*>& inputs,
                                        const std::vector<Tensor*>& outputs);
  Status callCutlassGemmTensorCore884(const std::vector<Tensor*>& inputs,
                                      const std::vector<Tensor*>& outputs);
  Status callCutlassGemmTensorCore(const std::vector<Tensor*>& inputs,
                                   const std::vector<Tensor*>& outputs);

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

  GemmTensor_F16_F16_Linear_AlignTensor_Sm70 mGemmF16F16LnSm70;
  GemmTensor_F16_F32_Linear_AlignTensor_Sm70 mGemmF16F32LnSm70;
  GemmCuda_F16_F16_Linear_AlignCuda mGemmCudaF16F16Ln;
  GemmCuda_F16_F32_Linear_AlignCuda mGemmCudaF16F32Ln;

  GemmTensor_F16_F16_Relu_AlignTensor_Sm70 mGemmF16F16ReluSm70;
  GemmTensor_F16_F32_Relu_AlignTensor_Sm70 mGemmF16F32ReluSm70;
  GemmCuda_F16_F16_Relu_AlignCuda mGemmCudaF16F16Relu;
  GemmCuda_F16_F32_Relu_AlignCuda mGemmCudaF16F32Relu;

  GemmTensor_F16_F16_Relu6_AlignTensor_Sm70 mGemmF16F16Relu6Sm70;
  GemmTensor_F16_F32_Relu6_AlignTensor_Sm70 mGemmF16F32Relu6Sm70;
  GemmCuda_F16_F16_Relu6_AlignCuda mGemmCudaF16F16Relu6;
  GemmCuda_F16_F32_Relu6_AlignCuda mGemmCudaF16F32Relu6;

  GemmTensor_F16_F16_Linear_AlignTensor_Sm75 mGemmF16F16LnSm75;
  GemmTensor_F16_F32_Linear_AlignTensor_Sm75 mGemmF16F32LnSm75;

  GemmTensor_F16_F16_Relu_AlignTensor_Sm75 mGemmF16F16ReluSm75;
  GemmTensor_F16_F32_Relu_AlignTensor_Sm75 mGemmF16F32ReluSm75;

  GemmTensor_F16_F16_Relu6_AlignTensor_Sm75 mGemmF16F16Relu6Sm75;
  GemmTensor_F16_F32_Relu6_AlignTensor_Sm75 mGemmF16F32Relu6Sm75;

  GemmCuda_F32_F32_Relu_AlignCuda mGemmCudaF32F32Relu;
  GemmCuda_F32_F32_Relu6_AlignCuda mGemmCudaF32F32Relu6;
  GemmCuda_F32_F32_Linear_AlignCuda mGemmCudaF32F32Ln;

  int mGpuComputeCap = 75;
  int mActivationType = 0;
  bool mFp16Infer = false;
  bool mFp32Infer = false;
  bool mFp16Fp32MixInfer = false;
  std::shared_ptr<Tensor> workspaceTensor;
  void* mWorkspace;
};

}  // namespace cuda
}  // namespace tars

#endif /* ConvCutlassExecution */
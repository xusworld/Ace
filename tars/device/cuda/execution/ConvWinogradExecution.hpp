//
//  ConvWinogradoperation.h
//  MNN
//
//  Created by MNN on 2022/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvWinogradExecution_hpp_
#define ConvWinogradExecution_hpp_

#include "ConvSingleInputoperation.h"
#include "CutlassGemmBatchedParam.hpp"
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"

namespace tars {
namespace cuda {

class ConvWinogradExecution : public Operation {
 public:
  struct Resource;
  static bool isValid(const Convolution2D* conv);
  ConvWinogradExecution(Device* backend, const tars::Op* op,
                        std::shared_ptr<Resource> res);
  virtual ~ConvWinogradExecution();

  struct Resource {
    Resource(Device* backend, const tars::Op* op);
    ~Resource();
    void* mFilter;
    void* mBias;
    std::shared_ptr<Tensor> weightTensor;
    std::shared_ptr<Tensor> biasTensor;
    KernelInfo mKernelInfo;
    Device* mBackend = nullptr;
    bool mUseHPack = false;
  };

  virtual Status onResize(const std::vector<Tensor*>& inputs,
                          const std::vector<Tensor*>& outputs) override;
  virtual Status onExecute(const std::vector<Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs) override;
  virtual bool onClone(Device* bn, const Op* op, Operation** dst) override;

 private:
  std::shared_ptr<Resource> mResource;
  const Op* mOp = nullptr;
  void* mBtdB_Buffer;
  void* mMatmul_Buffer;

  GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm75
      mGemmBatchedF16F16LnSm75;
  GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm75
      mGemmBatchedF16F32LnSm75;

  GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Column mGemmBatchedCudaF16F16Ln;
  GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Column mGemmBatchedCudaF16F32Ln;
  GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Column mGemmBatchedCudaF32F32Ln;

  std::shared_ptr<Tensor> workspaceTensor;
  void* mWorkspace;

  CutlassGemmInfo mGemmInfo;

  int mPadX;
  int mPadY;
  int mBlock2;
  int mGpuComputeCap;
  int mActivationType;
  bool mFp16Infer = false;
  bool mFp32Infer = false;
  bool mFp16Fp32MixInfer = false;
};

}  // namespace cuda
}  // namespace tars
#endif /* ConvWinogradExecution_hpp_ */
//
//  MatMuloperation.h
//  MNN
//
//  Created by MNN on 2020/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MatMulExecution_hpp
#define MatMulExecution_hpp

#include "CutlassGemmBatchedParam.hpp"
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {
class MatMulExecution : public Operation {
 public:
  MatMulExecution(bool transposeA, bool transposeB, Device *backend);
  virtual ~MatMulExecution();
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  void setArguments(const std::vector<Tensor *> &inputs,
                    const std::vector<Tensor *> &outputs);

 private:
  bool mTransposeA;
  bool mTransposeB;
  Device *mBackend = nullptr;

  std::shared_ptr<Tensor> mBiasTensor;
  GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm75
      mGemmBatchedF16F16LnAlign1RCSm75;
  GemmBatchedTensor_F32_F32_Linear_AlignCuda_Row_Column_Sm75
      mGemmBatchedF32F32LnAlign1RCSm75;
  GemmBatchedTensor_F16_F32_Linear_AlignCuda_Row_Column_Sm75
      mGemmBatchedF16F32LnAlign1RCSm75;

  GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm75
      mGemmBatchedF16F16LnAlign8RCSm75;
  GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Column_Sm75
      mGemmBatchedF32F32LnAlign8RCSm75;
  GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm75
      mGemmBatchedF16F32LnAlign8RCSm75;

  GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm75
      mGemmBatchedF16F16LnAlign8RRSm75;
  GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Row_Sm75
      mGemmBatchedF32F32LnAlign8RRSm75;
  GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Row_Sm75
      mGemmBatchedF16F32LnAlign8RRSm75;

  GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Column
      mGemmBatchedCudaF16F16LnAlign1RC;
  GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Column
      mGemmBatchedCudaF32F32LnAlign1RC;
  GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Column
      mGemmBatchedCudaF16F32LnAlign1RC;

  GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Row
      mGemmBatchedCudaF16F16LnAlign1RR;
  GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Row
      mGemmBatchedCudaF32F32LnAlign1RR;
  GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Row
      mGemmBatchedCudaF16F32LnAlign1RR;

  std::shared_ptr<Tensor> workspaceTensor;
  void *mWorkspace;
  void *mTempMatA;
  void *mTempMatB;
  void *mBiasPtr = nullptr;
  bool mNeedATempBuffer = false;
  bool mNeedBTempBuffer = false;
  bool mUseRRLayout = false;
  bool mResizeSetArgument = false;
  bool mNeedConvertMatAB = false;
  CutlassGemmInfo mGemmInfo;
  int mBatch = 1;
  int mGpuComputeCap;
  bool mFp16Infer = false;
  bool mFp32Infer = false;
  bool mFp16Fp32MixInfer = false;
};
}  // namespace cuda
}  // namespace tars

#endif

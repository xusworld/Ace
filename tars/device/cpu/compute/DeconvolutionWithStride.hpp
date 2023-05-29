//
//  DeconvolutionWithStride.hpp
//  MNN
//
//  Created by MNN on 2018/10/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DeconvolutionWithStride_hpp
#define DeconvolutionWithStride_hpp

#include <mutex>

#include "core/device.h"
#include "device/cpu/CPUDeconvolution.hpp"
namespace tars {
class DeconvolutionWithStride : public CPUDeconvolutionCommon {
 public:
  DeconvolutionWithStride(const Tensor *input, const Op *convOp, Device *b);
  virtual ~DeconvolutionWithStride();
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

  struct ComputeUnit {
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> dstBuffer;
    int xUnit = 0;
    int yUnit = 0;
    int xOffset = 0;
    int yOffset = 0;

    struct Winograd {
      std::shared_ptr<Tensor> dstTransformedBuffer;

      std::shared_ptr<Tensor> A;
      std::shared_ptr<Tensor> B;
      std::shared_ptr<Tensor> G;

      int srcUnitX = 0;
      int srcUnitY = 0;

      bool open = false;
    };

    Winograd winogradInfo;
  };

 private:
  bool _alloc(Device::StorageType type);
  void _release(Device::StorageType type);
  void _extract(const Op *convOp);

  std::shared_ptr<Tensor> mSrcBuffer;
  std::shared_ptr<Tensor> mMatMulPackBuffer;
  std::map<int, std::shared_ptr<Tensor>> mTransformedBuffer;
  std::shared_ptr<Tensor> mDestBuffer;

  std::vector<ComputeUnit> mComputeUnits;

  std::mutex mLock;
  int mStrideX = 1;
  int mStrideY = 1;
  std::vector<float> mPostParameters;
};
}  // namespace tars

#endif /* DeconvolutionWithStride_hpp */

//
//  DepthwiseConvInt8operation.h
//  MNN
//
//  Created by MNN on 2023/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DepthwiseConvInt8Execution_hpp
#define DepthwiseConvInt8Execution_hpp

#include "ConvInt8Cutlassoperation.h"
namespace tars {
namespace cuda {

class DepthwiseConvInt8Execution : public ConvInt8CutlassExecution {
 public:
  DepthwiseConvInt8Execution(Device *bn, const Op *op,
                             std::shared_ptr<Resource> resource);
  virtual ~DepthwiseConvInt8Execution();
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;
  virtual bool onClone(Device *bn, const Op *op, Operation **dst) override;

 private:
  const Op *mOp = nullptr;
  std::shared_ptr<ConvInt8CutlassExecution::Resource> mResource;
  std::pair<int, int> mPads;
  std::pair<int, int> mStrides;
  std::pair<int, int> mDilates;
  std::pair<int, int> mKernels;
  std::pair<int8_t, int8_t> mClamps;
};

}  // namespace cuda
}  // namespace tars

#endif /* DepthwiseConvInt8Execution_hpp */

//
//  MergeReluToConvolution.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
#include "MergeToConvolution.hpp"

using namespace tars;

class MergeReluToConvolution : public MergeToConvolution {
 public:
  bool merge2Convolution(const tars::OpT* inplaceOp,
                         tars::OpT* convolutionOp) const {
    if (inplaceOp->type == tars::OpType_ReLU &&
        inplaceOp->main.AsRelu()->slope == 0.0f) {
      convolutionOp->main.AsConvolution2D()->common->relu = true;
      return true;
    }
    return false;
  }

  bool merge2Convolution3D(const tars::OpT* inplaceOp,
                           tars::OpT* convolutionOp) const {
    if (inplaceOp->type == tars::OpType_ReLU &&
        inplaceOp->main.AsRelu()->slope == 0.0f) {
      convolutionOp->main.AsConvolution3D()->common->relu = true;
      return true;
    }
    return false;
  }
};
static PostConverterRegister<MergeReluToConvolution> __l(
    "MergeReluToConvolution");

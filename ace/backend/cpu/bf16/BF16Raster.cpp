//
//  BF16Raster.cpp
//  MNN
//
//  Created by MNN on 2020/5/25.
//  Copyright © 2018 Alibaba. All rights reserved.
//
#include "BF16Backend.hpp"
#include "backend/cpu/CPURaster.hpp"
namespace ace {
class BF16RasterFactory : public BF16Backend::BF16Creator {
 public:
  virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs,
                              const ace::Op* op,
                              Backend* backend) const override {
    if (outputs[0]->getType().code != halide_type_float) {
      return nullptr;
    }
    return new CPURaster(backend);
  }
};

REGISTER_BF16_OP_CREATOR(OpType_Raster, BF16RasterFactory);
}  // namespace ace

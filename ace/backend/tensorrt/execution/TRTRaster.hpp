//
//  TRTRaster.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTRaster_HPP
#define MNN_TRTRaster_HPP

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace ace {

class TRTRaster : public TRTCommonExecution {
 public:
  TRTRaster(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
            const std::vector<Tensor *> &outputs);
  virtual ~TRTRaster() = default;
  virtual std::vector<ITensor *> onEncode(
      const std::vector<ITensor *> &inputs) override;
};

}  // namespace ace

#endif  // MNN_TRTRaster_HPP

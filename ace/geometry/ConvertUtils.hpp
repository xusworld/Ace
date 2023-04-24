//
//  ConvertUtils.hpp
//  MNN
//
//  Created by MNN on 2020/04/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvertUtils_hpp
#define ConvertUtils_hpp
#include "core/TensorUtils.hpp"
#include "geometry/GeometryComputer.hpp"
namespace ace {
class ConvertUtils {
 public:
  static bool compute(Tensor* input, Tensor* output, CommandBuffer& res);
  static void broadcastto(Tensor* input, Tensor* output);
};
}  // namespace ace

#endif

//
//  Common.hpp
//  MNN
//
//  Created by MNN on 2020/07/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_CONVERTER_COMMON_COMMON_HPP_
#define MNN_CONVERTER_COMMON_COMMON_HPP_

#include "ace/HalideRuntime.h"
#include "ace/expr/Expr.hpp"
#include "ace_generated.h"

namespace ace {

DataType convertDataType(halide_type_t type);

DATA_FORMAT convertFormat(Express::Dimensionformat format);

}  // namespace ace

#endif  // MNN_CONVERTER_COMMON_COMMON_HPP_

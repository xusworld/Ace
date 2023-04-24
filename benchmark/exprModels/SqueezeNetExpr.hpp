//
//  SqueezeNetExpr.hpp
//  MNN
//  Reference paper: https://arxiv.org/pdf/1602.07360.pdf
//
//  Created by MNN on 2019/06/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SqueezeNetExpr_hpp
#define SqueezeNetExpr_hpp

#include <ace/expr/Expr.hpp>

ace::Express::VARP squeezeNetExpr(int numClass);

#endif  // SqueezeNetExpr_hpp

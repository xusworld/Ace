//
//  OnnxSoftplus.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <ace/expr/ExprCreator.hpp>

#include "OnnxExtraManager.hpp"
#include "ace_generated.h"

namespace ace {
namespace Express {

class OnnxSoftplusTransform : public OnnxExtraManager::Transform {
 public:
  virtual EXPRP onExecute(EXPRP expr) const override {
    auto input = expr->inputs()[0];
    auto newExpr = _Softplus(input)->expr().first;
    newExpr->setName(expr->name());
    return newExpr;
  }
};

static auto gRegister = []() {
  OnnxExtraManager::get()->insert(
      "Softplus",
      std::shared_ptr<OnnxExtraManager::Transform>(new OnnxSoftplusTransform));
  return true;
}();

}  // namespace Express
}  // namespace ace

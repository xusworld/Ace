//
//  OnnxSoftsign.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <ace/expr/Expr.hpp>

#include "OnnxExtraManager.hpp"
#include "ace_generated.h"

namespace ace {
namespace Express {

class OnnxSoftsignTransform : public OnnxExtraManager::Transform {
 public:
  virtual EXPRP onExecute(EXPRP expr) const override {
    auto input = expr->inputs()[0];
    auto newExpr = _Softsign(input)->expr().first;
    newExpr->setName(expr->name());
    return newExpr;
  }
};

static auto gRegister = []() {
  OnnxExtraManager::get()->insert(
      "Softsign",
      std::shared_ptr<OnnxExtraManager::Transform>(new OnnxSoftsignTransform));
  return true;
}();

}  // namespace Express
}  // namespace ace

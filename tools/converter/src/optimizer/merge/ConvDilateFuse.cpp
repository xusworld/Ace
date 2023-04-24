//
//  ConvDilateFuse.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "ace_generated.h"

namespace ace {
namespace Express {
static bool loadVar(VARP var, std::vector<int>& dst) {
  auto info = var->getInfo();
  auto ptr = var->readMap<int>();
  if (nullptr == info || nullptr == ptr) {
    return false;
  }
  dst.resize(info->size);
  ::memcpy(dst.data(), ptr, info->size * sizeof(int));
  return true;
}
static auto gRegister = []() {
  auto modify = [](EXPRP expr) {
    if (nullptr == expr->get()) {
      return false;
    }
    if (expr->get()->type() != OpType_BatchToSpaceND) {
      return false;
    }
    if (expr->outputs().size() > 1) {
      return false;
    }
    auto convInput = expr->inputs()[0];
    auto input1 = expr->inputs()[1];
    auto input2 = expr->inputs()[2];
    if (convInput->linkNumber() > 1) {
      return false;
    }
    auto convExpr = convInput->expr().first;
    if (nullptr == convExpr->get() ||
        convExpr->get()->main_type() != OpParameter_Convolution2D) {
      return false;
    }
    {
      auto convOp = convExpr->get();
      auto common = convOp->main_as_Convolution2D()->common();
      if (common->dilateX() > 1 || common->dilateY() > 1) {
        return false;
      }
      if (common->padMode() == PadMode_SAME) {
        return false;
      }
    }
    auto originInput = convExpr->inputs()[0];
    auto spaceToBatchExpr = originInput->expr().first;
    if (nullptr == spaceToBatchExpr->get() ||
        spaceToBatchExpr->get()->type() != OpType_SpaceToBatchND) {
      return false;
    }
    std::vector<int> outputBlockShape;
    if (!loadVar(input1, outputBlockShape)) {
      return false;
    }
    std::vector<int> outputPaddings;
    if (!loadVar(input2, outputPaddings)) {
      return false;
    }
    std::vector<int> inputBlockShape;
    if (!loadVar(spaceToBatchExpr->inputs()[1], inputBlockShape)) {
      return false;
    }
    std::vector<int> inputPaddings;
    if (!loadVar(spaceToBatchExpr->inputs()[2], inputPaddings)) {
      return false;
    }
    if (inputBlockShape.size() != outputBlockShape.size()) {
      return false;
    }
    for (int i = 0; i < inputBlockShape.size(); ++i) {
      if (inputBlockShape[i] != outputBlockShape[i]) {
        return false;
      }
    }
    if (inputPaddings.size() != outputPaddings.size()) {
      return false;
    }
    for (int i = 0; i < inputPaddings.size(); ++i) {
      inputPaddings[i] -= outputPaddings[i];
    }
    std::unique_ptr<OpT> newConv(convExpr->get()->UnPack());
    auto common = newConv->main.AsConvolution2D()->common.get();
    common->dilateY = inputBlockShape[0];
    if (inputBlockShape.size() > 1) {
      common->dilateX = inputBlockShape[1];
    }
    common->pads = inputPaddings;
    common->padMode = PadMode_CAFFE;
    auto newInputs = convExpr->inputs();
    newInputs[0] = spaceToBatchExpr->inputs()[0];
    auto newExpr = Expr::create(newConv.get(), newInputs, 1);
    newExpr->setName(convExpr->name());
    // Merge into convolution
    Expr::replace(expr, newExpr);
    return true;
  };
  TemplateMerge::getInstance("Merge").insertTemplateV2("ConvDilateFuse",
                                                       modify);
  return true;
}();
}  // namespace Express
}  // namespace ace

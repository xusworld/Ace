//
// Created by alibaba on 2019/9/11.
//

#include <core/TensorUtils.hpp>

#include "TRTBackend.hpp"
#include "TRTSoftmax.hpp"

using namespace std;

namespace ace {

TRTSoftmax::TRTSoftmax(Backend *b, const Op *op,
                       const std::vector<Tensor *> &inputs,
                       const std::vector<Tensor *> &outputs)
    : ace::TRTCommonExecution(b, op) {
  int axis = mOp->main_as_Axis()->axis();
  mAxis = axis < 0 ? axis + outputs[0]->dimensions() : axis;
}

std::vector<ITensor *> TRTSoftmax::onEncode(const std::vector<ITensor *> &xOp) {
  auto softmax_layer = mTrtBackend->getNetwork()->addSoftMax(*(xOp[0]));
  softmax_layer->setAxes(1U << mAxis);
  return {softmax_layer->getOutput(0)};
}

TRTCreatorRegister<TypedCreator<TRTSoftmax>> __softmax_op(OpType_Softmax);

}  // namespace ace

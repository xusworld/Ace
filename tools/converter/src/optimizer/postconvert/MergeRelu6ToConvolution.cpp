
#include "../PostTreatUtils.hpp"
#include "MergeToConvolution.hpp"

using namespace ace;

class MergeRelu6ToConvolution : public MergeToConvolution {
 public:
  bool merge2Convolution(const ace::OpT* inplaceOp,
                         ace::OpT* convolutionOp) const {
    if (inplaceOp->type == ace::OpType_ReLU6) {
      if (nullptr == inplaceOp->main.AsRelu6()) {
        convolutionOp->main.AsConvolution2D()->common->relu6 = true;
        return true;
      }
      if (inplaceOp->main.AsRelu6()->maxValue == 6.0f &&
          inplaceOp->main.AsRelu6()->minValue == 0.0f) {
        convolutionOp->main.AsConvolution2D()->common->relu6 = true;
        return true;
      }
    }
    return false;
  }

  bool merge2Convolution3D(const ace::OpT* inplaceOp,
                           ace::OpT* convolutionOp) const {
    if (inplaceOp->type == ace::OpType_ReLU6) {
      convolutionOp->main.AsConvolution3D()->common->relu6 = true;
      return true;
    }
    return false;
  }
};
static PostConverterRegister<MergeRelu6ToConvolution> __l(
    "MergeRelu6ToConvolution");

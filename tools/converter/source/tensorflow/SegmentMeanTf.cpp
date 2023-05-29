#include <string.h>

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(SegmentMeanTf);

tars::OpType SegmentMeanTf::opType() { return tars::OpType_Segment; }
tars::OpParameter SegmentMeanTf::type() {
  return tars::OpParameter_ReductionParam;
}

void SegmentMeanTf::run(tars::OpT *dstOp, TmpNode *srcNode) {
  dstOp->main.value = new tars::ReductionParamT;
  dstOp->main.AsReductionParam()->operation = tars::ReductionType_MEAN;
}

REGISTER_CONVERTER(SegmentMeanTf, SegmentMean);

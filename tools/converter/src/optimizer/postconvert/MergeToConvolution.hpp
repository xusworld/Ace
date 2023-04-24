//
//  MergeToConvolution.hpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
using namespace ace;

class MergeToConvolution : public PostConverter {
 public:
  virtual bool merge2Convolution(const ace::OpT* inplaceOp,
                                 ace::OpT* convolutionOp) const = 0;

  virtual bool merge2Convolution3D(const ace::OpT* inplaceOp,
                                   ace::OpT* convolutionOp) const = 0;

  virtual bool onExecute(std::unique_ptr<ace::NetT>& net) const override {
    // Merge Layer
    std::vector<ace::OpT*> readyToDelete;
    for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
      ace::OpT& currentOp = *(iter->get());
      if (currentOp.type != ace::OpType_Convolution &&
          currentOp.type != ace::OpType_Deconvolution &&
          currentOp.type != ace::OpType_ConvolutionDepthwise &&
          currentOp.type != ace::OpType_Convolution3D) {
        continue;
      }
      DCHECK(currentOp.outputIndexes.size() == 1) << "Conv output ERROR!";

      // merge Batchnorm/Relu/Relu6 to Convolution
      std::vector<ace::OpT*> nextOp = PostTreatUtils::_findOpByInputIndex(
          currentOp.outputIndexes[0], net.get());
      while (1) {
        if (nextOp.size() != 1) {
          break;
        }
        const int nextOutputIndex = nextOp[0]->outputIndexes[0];
        bool succ;
        if (currentOp.type == ace::OpType_Convolution3D) {
          succ = merge2Convolution3D(nextOp[0], &currentOp);
        } else {
          succ = merge2Convolution(nextOp[0], &currentOp);
        }
        if (PostTreatUtils::_isSingleInputOutput(nextOp[0]) && succ) {
          // LOG(INFO) << "Merge " << nextOp[0]->name.c_str()<< " into
          // convolution: " << currentOp.name.c_str();
          currentOp.outputIndexes[0] = nextOp[0]->outputIndexes[0];
          readyToDelete.push_back(nextOp[0]);
          nextOp =
              PostTreatUtils::_findOpByInputIndex(nextOutputIndex, net.get());
        } else {
          break;
        }
      }
    }
    for (auto op : readyToDelete) {
      PostTreatUtils::_removeOpInNet(op, net.get());
    }
    return true;
  }
};

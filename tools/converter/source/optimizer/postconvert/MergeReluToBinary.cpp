//
//  MergeReluToBinaryOp.hpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
using namespace tars;

class MergeReluToBinaryOp : public PostConverter {
 public:
  virtual bool onExecute(std::unique_ptr<tars::NetT>& net) const override {
    // Merge Layer
    std::vector<tars::OpT*> readyToDelete;
    for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
      tars::OpT& currentOp = *(iter->get());
      if (currentOp.type != tars::OpType_BinaryOp ||
          currentOp.inputIndexes.size() != 2) {
        continue;
      }
      DCHECK(currentOp.outputIndexes.size() == 1) << "Binary output ERROR!";

      // merge Relu/Relu6 to Binary
      std::vector<tars::OpT*> nextOp = PostTreatUtils::_findOpByInputIndex(
          currentOp.outputIndexes[0], net.get());
      while (1) {
        if (nextOp.size() != 1) {
          break;
        }
        const int nextOutputIndex = nextOp[0]->outputIndexes[0];

        bool nextRelu = (nextOp[0]->type == tars::OpType_ReLU &&
                         nextOp[0]->main.AsRelu()->slope == 0.0f);

        if (PostTreatUtils::_isSingleInputOutput(nextOp[0]) && nextRelu) {
          // LOG(INFO) << "Merge " << nextOp[0]->name.c_str()<< " into Binary:
          // ";
          //  currentOp.name.c_str();
          currentOp.main.AsBinaryOp()->activationType = 1;
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

static PostConverterRegister<MergeReluToBinaryOp> __l("MergeReluToBinaryOp");

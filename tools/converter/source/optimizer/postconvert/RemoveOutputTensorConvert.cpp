//
//  RemoveOutputTensorConvert.cpp
//  MNNConverter
//
//  Created by MNN on 2020/02/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
using namespace tars;
class RemoveOutputTensorConvert : public PostConverter {
 public:
  virtual bool onExecute(std::unique_ptr<tars::NetT>& net) const override {
    for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
      auto& op = *iter;
      if (op->outputIndexes.empty() || op->type != OpType_ConvertTensor) {
        iter++;
        continue;
      }
      if (PostTreatUtils::_findOpByInputIndex(op->outputIndexes[0], net.get())
              .size() > 0) {
        iter++;
        continue;
      }
      iter = net->oplists.erase(iter);
    }
    return true;
  }
};
static PostConverterRegister<RemoveOutputTensorConvert> __l(
    "RemoveOutputTensorConvert");

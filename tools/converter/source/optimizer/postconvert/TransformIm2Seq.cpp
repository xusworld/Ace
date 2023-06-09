//
//  TransformIm2Seq.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"

class TransformIm2Seq : public PostConverter {
 public:
  virtual bool onExecute(std::unique_ptr<tars::NetT>& net) const override {
    for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
      auto& op = *iter;
      if (op->type != tars::OpType_Im2Seq) {
        iter++;
        continue;
      }
      auto inputId = op->inputIndexes[0];
      auto outputId = op->outputIndexes[0];
      auto outputname = net->tensorName[outputId];

      // New Reshape
      tars::OpT* reshapeT = new tars::OpT;
      reshapeT->name = "____reshape____" + op->name;
      auto reshapeP = new tars::ReshapeT;
      reshapeP->dims.push_back(0);   // b
      reshapeP->dims.push_back(-1);  // c
      reshapeP->dims.push_back(1);   // h
      reshapeP->dims.push_back(0);   // w
      reshapeT->main.type = tars::OpParameter_Reshape;
      reshapeT->type = tars::OpType_Reshape;
      reshapeT->main.value = reshapeP;

      // Net Tensor
      net->tensorName.push_back(reshapeT->name);
      int tempId = net->tensorName.size() - 1;

      reshapeT->inputIndexes.push_back(inputId);
      reshapeT->outputIndexes.push_back(tempId);

      op->inputIndexes[0] = tempId;
      op->type = tars::OpType_Permute;

      auto convP = new tars::PermuteT;
      op->main.type = tars::OpParameter_Permute;
      op->main.value = convP;
      convP->dims.push_back(0);
      convP->dims.push_back(3);
      convP->dims.push_back(2);
      convP->dims.push_back(1);

      iter = net->oplists.insert(iter, std::unique_ptr<tars::OpT>(reshapeT));
    }
    return true;
  }
};
static PostConverterRegister<TransformIm2Seq> __l("TransformIm2Seq");

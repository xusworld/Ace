//
//  RemoveDropout.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <string>
#include <vector>

#include "../PostTreatUtils.hpp"
#include "RemoveTestNoUseOps.hpp"

using namespace tars;

class RemoveDropout : public RemoveTestNoUseOps {
 public:
  /* The Op's output set as input */
  bool shouldDeleteJudge(const tars::OpT* op,
                         const tars::NetT* const netPtr) const override {
    static auto unuseOpType = std::vector<OpType>({OpType_Dropout});
    static auto unuseExtraOpType = std::vector<std::string>({"Dropout"});
    if (std::find(unuseOpType.begin(), unuseOpType.end(), op->type) !=
        unuseOpType.end()) {
      return true;
    }
    if (op->type == OpType_Extra) {
      if (std::find(unuseExtraOpType.begin(), unuseExtraOpType.end(),
                    op->main.AsExtra()->type) != unuseExtraOpType.end()) {
        return true;
      }
      if (netPtr->sourceType == tars::NetSource_CAFFE &&
          op->main.AsExtra()->type == "Split") {
        return true;
      }
    }
    if (op->type == OpType_Cast) {
      if (op->main.AsCastParam()->dstT == op->main.AsCastParam()->srcT) {
        return true;
      }
      if (op->main.AsCastParam()->dstT == tars::DataType_DT_INT32 &&
          op->main.AsCastParam()->srcT == tars::DataType_DT_INT64) {
        return true;
      }
      if (op->main.AsCastParam()->srcT == tars::DataType_DT_INT32 &&
          op->main.AsCastParam()->dstT == tars::DataType_DT_INT64) {
        return true;
      }
    }
    return false;
  };
  bool shouldRemoveUnusefulInputs(const tars::OpT* op) const override {
    return false;
  };
  bool shouldDeleteOutput(const tars::OpT* op) const override { return false; };
};
static PostConverterRegister<RemoveDropout> __l("RemoveDropout");

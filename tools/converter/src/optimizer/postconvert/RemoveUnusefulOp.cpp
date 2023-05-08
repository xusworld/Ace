//
//  RemoveUnusefulOp.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <string>
#include <vector>

#include "../PostTreatUtils.hpp"
#include "RemoveTestNoUseOps.hpp"

using namespace ace;

class RemoveUnusefulOp : public RemoveTestNoUseOps {
 public:
  /* The Op's output set as input */
  bool shouldDeleteJudge(const ace::OpT* op,
                         const ace::NetT* const netPtr) const override {
    static auto unuseOpType = std::vector<OpType>({OpType_Seq2Out});
    static auto unuseExtraOpType = std::vector<std::string>(
        {"Identity", "IdentityN", "NoOp", "Assign", "Print", "Assert",
         "StopGradient", "Enter", "NextIteration"});

    // 判断 op 是否在待删除列表中
    if (std::find(unuseOpType.begin(), unuseOpType.end(), op->type) !=
        unuseOpType.end()) {
      return true;
    }

    if (op->type == OpType_Extra) {
      if (std::find(unuseExtraOpType.begin(), unuseExtraOpType.end(),
                    op->main.AsExtra()->type) != unuseExtraOpType.end()) {
        return true;
      }
      if (netPtr->sourceType == ace::FrontendFramework_CAFFE &&
          op->main.AsExtra()->type == "Split") {
        return true;
      }
    }

    // 框架内部特殊的处理逻辑
    if (op->type == OpType_Cast) {
      if (op->main.AsCastParam()->dstT == op->main.AsCastParam()->srcT) {
        return true;
      }
      if (op->main.AsCastParam()->dstT == ace::DataType_DT_INT32 &&
          op->main.AsCastParam()->srcT == ace::DataType_DT_INT64) {
        return true;
      }
      if (op->main.AsCastParam()->srcT == ace::DataType_DT_INT32 &&
          op->main.AsCastParam()->dstT == ace::DataType_DT_INT64) {
        return true;
      }
    }
    // 如果 Op 是 Concat O, 但是 op 的输入只有一个，则需删除
    if (op->type == OpType_Concat && op->inputIndexes.size() == 1) {
      return true;
    }
    // 如果 Op 是 Slice Op, 但是 op 是输出只有一个，则需删除
    if (op->type == OpType_Slice && op->outputIndexes.size() == 1) {
      return true;
    }
    return false;
  };

  bool shouldRemoveUnusefulInputs(const ace::OpT* op) const override {
    if (op->type == OpType_Extra) {
      if (op->main.AsExtra()->type == "Assert") {
        return true;
      }
      if (op->main.AsExtra()->type == "NoOp") {
        return true;
      }
      if (op->main.AsExtra()->type == "Print") {
        return true;
      }
      // StopGradient should be replaced by Identity.
      // if (op->main.AsExtra()->type == "StopGradient") {
      //     return true;
      // }
    }
    return false;
  };

  bool shouldDeleteOutput(const ace::OpT* op) const override {
    if (op->type == OpType_Extra) {
      return op->main.AsExtra()->type == "Assert";
    }
    return false;
  };
};
static PostConverterRegister<RemoveUnusefulOp> __l("RemoveUnusefulOp");

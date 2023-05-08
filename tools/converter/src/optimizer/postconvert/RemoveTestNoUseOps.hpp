//
//  RemoveTestNoUseOps.hpp
//  MNNConverter
//
//  Created by MNN on 2019/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../PostTreatUtils.hpp"
#include "ace/MNNDefine.h"

using namespace ace;

class RemoveTestNoUseOps : public PostConverter {
 public:
  /* The Op's output set as input */
  virtual bool shouldDeleteJudge(const ace::OpT* op,
                                 const ace::NetT* const netPtr) const = 0;

  virtual bool shouldRemoveUnusefulInputs(const ace::OpT* op) const = 0;

  virtual bool shouldDeleteOutput(const ace::OpT* op) const = 0;

  // 删除 useless op 的执行逻辑
  virtual bool onExecute(std::unique_ptr<ace::NetT>& net) const override {
    const ace::NetT* const netPtr = net.get();

    std::unordered_set<int> removedInputs;
    for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
      // 获取当前 op
      auto& op = *iter;
      // 判断是否删除当前 op
      bool shouldDelete = shouldDeleteJudge(op.get(), netPtr);
      if (!shouldDelete) {
        iter++;
        continue;
      }
      // 1. 判断是否删除 output
      bool deleteOutput = shouldDeleteOutput(op.get());
      // 如果当前 op 的输入或者输入为空，则直接从 op 列表中删除
      if (op->outputIndexes.empty() || op->inputIndexes.empty()) {
        iter = net->oplists.erase(iter);
        continue;
      }
      // op 的第一个输入
      auto originInput = op->inputIndexes[0];
      // op 的所有输出
      auto originOutputs = op->outputIndexes;
      // 遍历计算图上所有的 op 列表
      for (auto subIter = net->oplists.begin(); subIter != net->oplists.end();
           subIter++) {
        auto& subOp = *subIter;

        if (deleteOutput) {  // 删除 op 所有的输出
          for (auto iter = subOp->inputIndexes.begin();
               iter != subOp->inputIndexes.end();) {
            // 判断遍历节点的输入是不是在当前节点的输出中，如果是，则删除该输出
            if (std::find(originOutputs.begin(), originOutputs.end(), *iter) !=
                originOutputs.end()) {
              iter = subOp->inputIndexes.erase(iter);
              continue;
            }
            iter++;
          }
        } else {  // 不删除 op 的输出
          for (int v = 0; v < subOp->inputIndexes.size(); ++v) {
            // 如果遍历节点的某个输入在待删除节点的输出中，则将待删除节点的输入作为遍历节点的输入
            if (std::find(originOutputs.begin(), originOutputs.end(),
                          subOp->inputIndexes[v]) != originOutputs.end()) {
              subOp->inputIndexes[v] = originInput;
            }
          }
        }
      }

      // 2. 判断是否删除输入
      bool removeUselessInput = shouldRemoveUnusefulInputs(op.get());
      if (removeUselessInput) {
        for (int input : op->inputIndexes) {
          removedInputs.emplace(input);
        }
      }
      iter = net->oplists.erase(iter);
    }

    // Remove the op only if the reference counts of it's all outputs
    // are reduced to be zero.
    std::unordered_map<int, int /*reference count*/> uselessIndex;
    for (const auto& op : net->oplists) {
      for (int input : op->inputIndexes) {
        auto it = uselessIndex.find(input);
        if (it == uselessIndex.end()) {
          uselessIndex.emplace(input, 1);
        } else {
          ++it->second;
        }
      }
    }

    // Set reference count 1 for all net outputs.
    for (const auto& op : net->oplists) {
      for (int output : op->outputIndexes) {
        auto it = uselessIndex.find(output);
        if (it == uselessIndex.end()) {
          if (removedInputs.count(output)) {
            uselessIndex.emplace(output, 0);
          } else {
            uselessIndex.emplace(output, 1);
          }
        }
      }
    }

    bool needIteration = false;
    do {
      needIteration = false;
      for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
        auto& op = *iter;
        bool useless = true;
        for (auto index : op->outputIndexes) {
          if (uselessIndex.at(index) > 0) {
            useless = false;
            break;
          }
        }
        if (!useless) {
          iter++;
          continue;
        }
        if (!op->inputIndexes.empty()) {
          for (auto index : op->inputIndexes) {
            auto it = uselessIndex.find(index);
            MNN_ASSERT(it != uselessIndex.end());
            --it->second;
          }
          needIteration = true;
        }
        iter = net->oplists.erase(iter);
      }
    } while (needIteration);

    return true;
  }
};

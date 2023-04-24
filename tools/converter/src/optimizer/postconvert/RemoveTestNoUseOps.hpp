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

  virtual bool onExecute(std::unique_ptr<ace::NetT>& net) const override {
    const ace::NetT* const netPtr = net.get();

    std::unordered_set<int> removedInputs;
    for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
      auto& op = *iter;
      bool shouldDelete = shouldDeleteJudge(op.get(), netPtr);
      if (!shouldDelete) {
        iter++;
        continue;
      }
      bool deleteOutput = shouldDeleteOutput(op.get());
      // Find the next op
      if (op->outputIndexes.empty() || op->inputIndexes.empty()) {
        iter = net->oplists.erase(iter);
        continue;
      }

      auto originInput = op->inputIndexes[0];
      auto originOutputs = op->outputIndexes;
      for (auto subIter = net->oplists.begin(); subIter != net->oplists.end();
           subIter++) {
        auto& subOp = *subIter;
        if (deleteOutput) {
          for (auto iter = subOp->inputIndexes.begin();
               iter != subOp->inputIndexes.end();) {
            if (std::find(originOutputs.begin(), originOutputs.end(), *iter) !=
                originOutputs.end()) {
              iter = subOp->inputIndexes.erase(iter);
              continue;
            }
            iter++;
          }
        } else {
          for (int v = 0; v < subOp->inputIndexes.size(); ++v) {
            if (std::find(originOutputs.begin(), originOutputs.end(),
                          subOp->inputIndexes[v]) != originOutputs.end()) {
              subOp->inputIndexes[v] = originInput;
            }
          }
        }
      }
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
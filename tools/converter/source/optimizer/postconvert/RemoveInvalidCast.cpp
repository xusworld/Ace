//
//  RemoveInvalidCast.cpp
//  MNNConverter
//
//  Created by MNN on 2021/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../PostTreatUtils.hpp"
using namespace tars;
class RemoveInvalidCast : public PostConverter {
 public:
  static bool outputBool(int operation) {
    if (operation == BinaryOpOperation_GREATER_EQUAL) {
      return true;
    }
    if (operation == BinaryOpOperation_GREATER) {
      return true;
    }
    if (operation == BinaryOpOperation_LESS) {
      return true;
    }
    if (operation == BinaryOpOperation_LESS_EQUAL) {
      return true;
    }
    if (operation == BinaryOpOperation_EQUAL) {
      return true;
    }
    if (operation == BinaryOpOperation_NOTEQUAL) {
      return true;
    }
    return false;
  }
  virtual bool onExecute(std::unique_ptr<tars::NetT>& net) const override {
    if (net->sourceType == tars::NetSource_TENSORFLOW ||
        net->sourceType == tars::NetSource_TFLITE) {
      // The two framework has valid src type for cast, don't need treat
      return true;
    }
    if (net->sourceType == tars::NetSource_CAFFE) {
      // For caffe has no invalid cast op
      return true;
    }
    bool needTreat = false;
    for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
      auto& op = *iter;
      if (op->type == tars::OpType_Cast) {
        needTreat = true;
        break;
      }
    }
    if (!needTreat) {
      return true;
    }
    // Infer DataType for All Tensor
    std::vector<tars::DataType> types(net->tensorName.size(),
                                      tars::DataType_DT_INVALID);
    for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
      auto& op = *iter;
      switch (op->type) {
        case tars::OpType_Input:
          types[op->outputIndexes[0]] = op->main.AsInput()->dtype;
          break;
        case tars::OpType_Cast:
          types[op->outputIndexes[0]] = op->main.AsCastParam()->dstT;
          break;
        case tars::OpType_Const:
        case tars::OpType_TrainableParam:
          types[op->outputIndexes[0]] = op->main.AsBlob()->dataType;
          break;
        case tars::OpType_Fill:
          types[op->outputIndexes[0]] = types[op->inputIndexes[1]];
          break;
        case tars::OpType_Shape:
        case tars::OpType_Size:
        case tars::OpType_Rank:
        case tars::OpType_UnravelIndex:
          types[op->outputIndexes[0]] = tars::DataType_DT_INT32;
          break;
        case tars::OpType_RandomUniform:
          types[op->outputIndexes[0]] = op->main.AsRandomUniform()->type;
          break;
        case tars::OpType_ArgMax:
          types[op->outputIndexes[0]] = tars::DataType_DT_INT32;
          break;
        case tars::OpType_TopKV2:
          types[op->outputIndexes[0]] = types[op->inputIndexes[0]];
          if (op->outputIndexes.size() > 1) {
            types[op->outputIndexes[1]] = tars::DataType_DT_INT32;
          }
          break;
        case tars::OpType_ScatterNd:
        case tars::OpType_Select:
          types[op->outputIndexes[0]] = types[op->inputIndexes[1]];
          break;
        case tars::OpType_OneHot:
          types[op->outputIndexes[0]] = types[op->inputIndexes[2]];
          break;
        case tars::OpType_Extra:
        case tars::OpType_Plugin:
          break;
        case tars::OpType_BinaryOp: {
          if (outputBool(op->main.AsBinaryOp()->opType)) {
            types[op->outputIndexes[0]] = DataType_DT_BOOL;
          } else {
            types[op->outputIndexes[0]] = types[op->inputIndexes[0]];
          }
        } break;
        default:
          if (op->inputIndexes.size() > 0) {
            for (int i = 0; i < op->outputIndexes.size(); ++i) {
              types[op->outputIndexes[i]] = types[op->inputIndexes[0]];
            }
          }
          break;
      }
    }
    // Remove Useless Cast
    const tars::NetT* const netPtr = net.get();
    for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
      auto& op = *iter;
      if (op->type != tars::OpType_Cast) {
        iter++;
        continue;
      }
      if (types[op->inputIndexes[0]] == tars::DataType_DT_INVALID) {
        iter++;
        continue;
      }
      if (types[op->inputIndexes[0]] != types[op->outputIndexes[0]]) {
        iter++;
        break;
      }
      if (std::find(net->outputName.begin(), net->outputName.end(),
                    net->tensorName[op->outputIndexes[0]]) !=
          net->outputName.end()) {
        iter++;
        continue;
      }
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
        for (int v = 0; v < subOp->inputIndexes.size(); ++v) {
          if (std::find(originOutputs.begin(), originOutputs.end(),
                        subOp->inputIndexes[v]) != originOutputs.end()) {
            subOp->inputIndexes[v] = originInput;
          }
        }
      }
      iter = net->oplists.erase(iter);
    }
    return true;
  }
};
static PostConverterRegister<RemoveInvalidCast> __l("RemoveInvalidCast");

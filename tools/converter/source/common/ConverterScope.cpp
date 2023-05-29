//
//  ConverterScope.cpp
//  MNNConverter
//
//  Created by MNN on 2021/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>

#include "ConverterScope.hpp"

ConverterScope::ConverterScope()
    : mNet(nullptr), mSubNet(nullptr), mParent(nullptr) {}
ConverterScope::ConverterScope(tars::NetT* net)
    : mNet(net), mSubNet(nullptr), mParent(nullptr) {}
ConverterScope::ConverterScope(tars::SubGraphProtoT* subnet,
                               tars::NetT* parentNet,
                               ConverterScope* parentScope)
    : mNet(parentNet), mSubNet(subnet), mParent(parentScope) {}

std::vector<std::string>& ConverterScope::tensors() {
  return mSubNet ? mSubNet->tensors : mNet->tensorName;
}

std::vector<std::unique_ptr<tars::OpT>>& ConverterScope::oplists() {
  return mSubNet ? mSubNet->nodes : mNet->oplists;
}

std::vector<std::string>& ConverterScope::deps() {
  return mParent ? mParent->subgraphDeps : this->subgraphDeps;
}

int ConverterScope::declareTensor(std::string name) {
  auto iter = mTensorIdx.find(name);
  if (iter != mTensorIdx.end()) {
    return iter->second;
  }
  tensors().push_back(name);
  int newIdx = mTensorIdx.size();
  mTensorIdx.insert(std::make_pair(name, newIdx));
  return newIdx;
}

std::string ConverterScope::lookupTensorByIdx(int idx) {
  if (idx < tensors().size()) {
    return tensors()[idx];
  }
  return "NaN";
}

int ConverterScope::buildIntConstOp(std::vector<int> data, std::string name) {
  int idx = declareTensor(name);
  std::unique_ptr<tars::OpT> constOp(new tars::OpT);
  constOp->name = name;
  constOp->type = tars::OpType_Const;
  constOp->main.type = tars::OpParameter_Blob;
  auto blob = new tars::BlobT;
  blob->dims = {static_cast<int>(data.size())};
  blob->dataType = tars::DataType_DT_INT32;
  blob->int32s = data;
  blob->dataFormat = tars::MNN_DATA_FORMAT_NCHW;
  constOp->main.value = blob;
  constOp->outputIndexes.push_back(idx);
  oplists().emplace_back(std::move(constOp));
  return idx;
}

int ConverterScope::buildIntInputOp(std::string name) {
  int idx = declareTensor(name);
  std::unique_ptr<tars::OpT> inputOp(new tars::OpT);
  inputOp->name = name;
  inputOp->type = tars::OpType_Input;
  inputOp->main.type = tars::OpParameter_Input;
  auto param = new tars::InputT;
  param->dtype = tars::DataType_DT_INT32;
  param->dformat = tars::MNN_DATA_FORMAT_NCHW;
  inputOp->main.value = param;
  inputOp->outputIndexes.push_back(idx);
  if (mSubNet) {
    mSubNet->inputs.push_back(idx);
  }
  oplists().emplace_back(std::move(inputOp));
  return idx;
}

void ConverterScope::addInputForOp(tars::OpT* op, std::string inputName,
                                   bool allowSameInput) {
  int idx = this->lookupTensor(inputName);
  if (idx < 0) {
    idx = this->buildIntInputOp(inputName);
    if (mParent) {
      mParent->subgraphDeps.push_back(inputName);
    }
  }
  if (allowSameInput ||
      std::find(op->inputIndexes.begin(), op->inputIndexes.end(), idx) ==
          op->inputIndexes.end()) {
    op->inputIndexes.push_back(idx);
  }
}

void ConverterScope::dealSubgraphDeps() {
  if (!mSubNet) {
    return;
  }
  for (const auto& dep : subgraphDeps) {
    int idx = this->lookupTensor(dep);
    if (idx < 0) {
      idx = this->buildIntInputOp(dep);
      if (mParent) {
        mParent->subgraphDeps.push_back(dep);
      }
    }
    if (std::find(mSubNet->inputs.begin(), mSubNet->inputs.end(), idx) ==
        mSubNet->inputs.end()) {
      mSubNet->inputs.push_back(idx);
    }
  }
}

void ConverterScope::dealSubgraphDepsForOp(tars::OpT* op) {
  for (const auto& dep : subgraphDeps) {
    addInputForOp(op, dep);
  }
}

void ConverterScope::buildCondGraph(const std::string& name,
                                    const std::string& iName,
                                    const std::string& mName,
                                    const std::string& kName) {
  // declare i < M && keep_going
  std::unique_ptr<tars::SubGraphProtoT> subgraph(new tars::SubGraphProtoT);
  subgraph->name = name;
  std::unique_ptr<ConverterScope> scope(
      new ConverterScope(subgraph.get(), mNet, this));
  int idxI = scope->buildIntInputOp(iName);
  int idxM = scope->buildIntInputOp(mName);
  int idxK = scope->buildIntInputOp(kName);
  int idxC = scope->declareTensor(name + "/compare_res");
  int idxO = scope->declareTensor(name + "/keepgoing_res");
  // i < M
  tars::OpT* compareOp = new tars::OpT;
  compareOp->name = name + "/compare";
  compareOp->type = tars::OpType_BinaryOp;
  compareOp->main.type = tars::OpParameter_BinaryOp;
  auto param = new tars::BinaryOpT;
  param->opType = tars::BinaryOpOperation_LESS;
  param->T = tars::DataType_DT_INT32;
  compareOp->main.value = param;
  compareOp->inputIndexes.resize(2);
  compareOp->inputIndexes[0] = idxI;
  compareOp->inputIndexes[1] = idxM;
  compareOp->outputIndexes.push_back(idxC);
  subgraph->nodes.emplace_back(compareOp);
  // keep_going
  tars::OpT* keepOp = new tars::OpT;
  keepOp->name = name + "/keepgoing";
  keepOp->type = tars::OpType_BinaryOp;
  keepOp->main.type = tars::OpParameter_BinaryOp;
  param = new tars::BinaryOpT;
  param->opType = tars::BinaryOpOperation_MUL;
  param->T = tars::DataType_DT_INT32;
  keepOp->main.value = param;
  keepOp->inputIndexes.resize(2);
  keepOp->inputIndexes[0] = idxC;
  keepOp->inputIndexes[1] = idxK;
  keepOp->outputIndexes.push_back(idxO);
  subgraph->nodes.emplace_back(keepOp);
  // cond_res
  subgraph->outputs.push_back(idxO);
  mNet->subgraphs.emplace_back(std::move(subgraph));
}

void ConverterScope::buildIncrement(std::string name, std::string iName) {
  // for while_body: i++
  int idxOne = buildIntConstOp({1}, name + "/increment_1");
  int idxInc = declareTensor(name + "/increment_i");
  tars::OpT* incrementOp = new tars::OpT;
  incrementOp->name = name + "/increment";
  incrementOp->type = tars::OpType_BinaryOp;
  incrementOp->main.type = tars::OpParameter_BinaryOp;
  auto param = new tars::BinaryOpT;
  param->opType = tars::BinaryOpOperation_ADD;
  param->T = tars::DataType_DT_INT32;
  incrementOp->main.value = param;
  addInputForOp(incrementOp, iName);
  incrementOp->inputIndexes.push_back(idxOne);
  incrementOp->outputIndexes.push_back(idxInc);
  oplists().emplace_back(incrementOp);
  mSubNet->outputs.push_back(idxInc);
}

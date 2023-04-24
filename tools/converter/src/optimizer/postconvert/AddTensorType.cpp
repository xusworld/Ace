//
//  AddTensorType.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
using namespace ace;
class AddTensorType : public PostConverter {
 public:
  virtual bool onExecute(std::unique_ptr<ace::NetT>& net) const override {
    auto& mNet = net;
    for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end();
         iter++) {
      auto& op = *iter;
      if (op->type == ace::OpType_StridedSlice) {
        auto parameter = op->main.AsStridedSliceParam();
        auto dataType = parameter->T;

        {
          int index = op->inputIndexes[0];
          auto describe =
              std::unique_ptr<ace::TensorDescribeT>(new ace::TensorDescribeT);
          describe->index = index;
          describe->blob = std::unique_ptr<ace::BlobT>(new ace::BlobT);
          describe->blob->dataType = dataType;
          mNet->extraTensorDescribe.push_back(std::move(describe));
        }
        {
          int index = op->outputIndexes[0];
          auto describe =
              std::unique_ptr<ace::TensorDescribeT>(new ace::TensorDescribeT);
          describe->index = index;
          describe->blob = std::unique_ptr<ace::BlobT>(new ace::BlobT);
          describe->blob->dataType = dataType;
          mNet->extraTensorDescribe.push_back(std::move(describe));
        }
      }
      if (op->type == ace::OpType_Const) {
        auto constP = op->main.AsBlob();
        {
          int index = op->outputIndexes[0];
          auto describe =
              std::unique_ptr<ace::TensorDescribeT>(new ace::TensorDescribeT);
          describe->index = index;
          describe->blob = std::unique_ptr<ace::BlobT>(new ace::BlobT);
          describe->blob->dataType = constP->dataType;
          mNet->extraTensorDescribe.push_back(std::move(describe));
        }
      }
    }
    return true;
  }
};
static PostConverterRegister<AddTensorType> __l("AddTensorType");

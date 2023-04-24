//
//  TransformBatchNormal.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"

class TransformBatchNormal : public PostConverter {
 public:
  virtual bool onExecute(std::unique_ptr<ace::NetT>& net) const override {
    for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
      auto& op = *iter;
      const ace::OpType opType = op->type;

      if (ace::OpType_BatchNorm != opType) {
        iter++;
        continue;
      }
      const int inputSize = op->inputIndexes.size();
      DCHECK(inputSize == 1 || inputSize == 3)
          << "MNN BatchnNorm input size error!";
      // instance norm have three input tensors(input_tensor, mean, variance)
      if (inputSize == 3) {
        iter++;
        continue;
      }
      // DLOG(INFO) << "change BatchNorm to Scale: " << op->name;
      auto batchnormParam = op->main.AsBatchNorm();
      auto scaleParam = new ace::ScaleT;
      scaleParam->channels = batchnormParam->channels;
      scaleParam->scaleData.resize(batchnormParam->channels);
      scaleParam->biasData.resize(batchnormParam->channels);
      const float* slopePtr = batchnormParam->slopeData.data();
      const float* meanDataPtr = batchnormParam->meanData.data();
      const float* varDataPtr = batchnormParam->varData.data();
      const float* biasDataPtr = batchnormParam->biasData.data();
      const float eps = batchnormParam->epsilon;

      for (int i = 0; i < batchnormParam->channels; i++) {
        float sqrt_var = sqrt(varDataPtr[i] + eps);
        scaleParam->biasData[i] =
            biasDataPtr[i] - slopePtr[i] * meanDataPtr[i] / sqrt_var;
        scaleParam->scaleData[i] = slopePtr[i] / sqrt_var;
      }

      op->type = ace::OpType_Scale;
      op->main.type = ace::OpParameter_Scale;
      op->main.value = scaleParam;
    }
    return true;
  }
};
static PostConverterRegister<TransformBatchNormal> __l("TransformBatchNormal");

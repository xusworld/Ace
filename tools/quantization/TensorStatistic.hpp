//
//  TensorStatistic.hpp
//  MNN
//
//  Created by MNN on 2019/06/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <memory>
#include <string>
#include <vector>

#include "core/tensor.h"

enum GET_THRESHOLD_METHOD {
  THRESHOLD_MAX = 0,
  THRESHOLD_KL = 1,
};

class TensorStatistic {
 public:
  TensorStatistic(const tars::Tensor* tensor, std::string method,
                  const std::string& name, float featureClampValue,
                  int binNumber = 2048,
                  GET_THRESHOLD_METHOD thresholdMethod = THRESHOLD_KL);
  ~TensorStatistic() {
    // Do nothing
  }

  void resetUpdatedDistributionFlag() { mUpdatedDistributionFlag = false; }
  void resetUpdatedRangeFlags() { mUpdatedRangeFlags = false; }
  void updateRange();
  void resetDistribution();
  void updateDistribution();

  void setThresholdMethod(GET_THRESHOLD_METHOD thresholdMethod);

  float finishAndCompute();

  // only this one for ADMM
  float computeScaleADMM();

  std::string name() { return mName; }

  bool visited() { return mVisited; }

  void setVisited(bool visited) { mVisited = visited; }

  std::pair<std::vector<float>, float> fakeQuantFeature();
  float computeDistance(std::vector<float> fakeQuantedFeature);

 private:
  int _computeThreshold(const std::vector<float>& distribution);
  // <minVal, maxVal> for every channel for the Tensor
  std::pair<float, float> mRange;
  // mBinNumber / maxValue: the number of bin for range 1
  float mInterval;
  // if the i-th channel's maxValue > 0.00001f, mValidChannel[i] is true
  bool mValid;
  // [c * mBinNumber]: store every channel's distribution using bin
  std::vector<float> mDistribution;

  std::shared_ptr<tars::Tensor> mHostTensor;
  // the Tensor
  const tars::Tensor* mOriginTensor;
  // bin number for distribution
  int mBinNumber;
  // has update or not, assert update once
  bool mUpdatedDistributionFlag = false;
  bool mUpdatedRangeFlags = false;

  std::string mName;
  GET_THRESHOLD_METHOD mThresholdMethod = THRESHOLD_KL;
  bool mVisited = false;
  float mScale;
  float mFeatureClampValue = 127.0f;
};

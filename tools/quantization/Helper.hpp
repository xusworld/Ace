//
//  Helper.hpp
//  MNN
//
//  Created by MNN on 2019/08/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/ImageProcess.hpp>
#include <set>
#include <string>

#include "MNN_generated.h"
#include "core/tensor.h"
#include "logkit.h"

#pragma once
class Helper {
 public:
  struct PreprocessConfig {
    int targetHeight;
    int targetWidth;
    float centerCropHeight = 1.0;
    float centerCropWidth = 1.0;
  };

  enum InputType {
    IMAGE = 0,
    SEQUENCE = 1,
  };

  static std::set<std::string> gNotNeedFeatureOp;

  static std::set<tars::OpType> INT8SUPPORTED_OPS;

  static std::set<std::string> featureQuantizeMethod;
  static std::set<std::string> weightQuantizeMethod;

  static bool fileExist(const std::string& file);
  static void readClibrationFiles(std::vector<std::string>& images,
                                  const std::string& filePath,
                                  int* usedImageNum);
  static void preprocessInput(tars::CV::ImageProcess* pretreat,
                              PreprocessConfig PreprocessConfig,
                              const std::string& filename, tars::Tensor* input,
                              InputType inputType);
  static void invertData(float* dst, const float* src, int size);
  static bool stringEndWith(std::string const& fullString,
                            std::string const& ending);
};

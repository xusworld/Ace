//
//  PostTreatUtils.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef POSTTREATUTILS_HPP
#define POSTTREATUTILS_HPP

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>

#include "MNN_generated.h"
#include "logkit.h"
class PostConverter {
 public:
  PostConverter() = default;
  virtual ~PostConverter() = default;
  virtual bool onExecute(std::unique_ptr<tars::NetT>& net) const = 0;
  static PostConverter* get(std::string key);
  static void add(std::shared_ptr<PostConverter> converter, std::string key);

 private:
  static std::map<std::string, std::shared_ptr<PostConverter>>* getConvertMap();
};

template <class T>
class PostConverterRegister {
 public:
  PostConverterRegister(const char* claim) {
    T* instance = new T;
    PostConverter::add(std::shared_ptr<PostConverter>(instance), claim);
  }
};

class PostTreatUtils {
 public:
  static tars::OpT* _findOpByOutputIndex(int outputIndex,
                                         const tars::NetT* net);
  static std::vector<tars::OpT*> _findOpByInputIndex(int inputIndex,
                                                     const tars::NetT* net);
  static void _removeOpInNet(tars::OpT* op, tars::NetT* net);
  static bool _isSingleInputOutput(const tars::OpT* op);

  static int _getOpDecestorCount(tars::OpT* op, const tars::NetT* net);
  static bool _replace(std::vector<int>& indexes, int freshIndex, int oldIndex);

 private:
  PostTreatUtils();
};

#endif  // POSTTREATUTILS_HPP

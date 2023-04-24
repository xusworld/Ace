//
//  revertMNNModel.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef REVERTMNNMODEL_HPP
#define REVERTMNNMODEL_HPP

#include "ace_generated.h"

class Revert {
 public:
  Revert(const char* originalModelFileName);
  ~Revert();
  void* getBuffer() const;
  const size_t getBufferSize() const;
  void initialize(float sparsity = 0.0f, int sparseBlockOC = 1);
  static void fillRandValue(float* data, size_t size);

 private:
  Revert();
  std::unique_ptr<ace::NetT> mMNNNet;
  size_t mBufferSize;
  std::shared_ptr<uint8_t> mBuffer;
  void randStart();
  void packMNNNet();
};

#endif  // REVERTMNNMODEL_HPP

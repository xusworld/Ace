//
//  Operation.cpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/operation.h"

namespace tars {

const Operation::Creator* Operation::searchExtraCreator(const std::string& key,
                                                        MNNForwardType type) {
  // Depercerate
  return nullptr;
}

bool Operation::insertExtraCreator(std::shared_ptr<Creator> creator,
                                   const std::string& key,
                                   MNNForwardType type) {
  // Depercerate
  return true;
}

bool Operation::removeExtraCreator(const std::string& key,
                                   MNNForwardType type) {
  // Depercerate
  return true;
}
}  // namespace tars

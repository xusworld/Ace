//
//  tensorflowConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TENSORFLOWCONVERTER_HPP
#define TENSORFLOWCONVERTER_HPP

#include <MNN/MNNDefine.h>

#include <string>

#include "MNN_generated.h"
/**
 * @brief convert tensorflow model to MNN model
 * @param inputModel tensorflow model name(xx.pb)
 * @param bizCode(not used, always is MNN)
 * @param MNN net
 */
MNN_PUBLIC int tensorflow2MNNNet(const std::string inputModel,
                                 const std::string bizCode,
                                 std::unique_ptr<tars::NetT>& netT);

#endif  // TENSORFLOWCONVERTER_HPP

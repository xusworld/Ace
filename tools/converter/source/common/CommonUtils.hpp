//
//  CommonUtils.hpp
//  MNNConverter
//
//  Created by MNN on 2021/08/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef COMMMON_UTILS_HPP
#define COMMMON_UTILS_HPP

#include <MNN/MNNDefine.h>

#include <map>

#include "MNN_compression.pb.h"
#include "MNN_generated.h"
#include "config.hpp"

void converToStaticModel(const tars::Net* net,
                         std::map<std::string, std::vector<int>>& inputConfig,
                         std::string mnnFile);
void removeParams(std::unique_ptr<tars::NetT>& netT);
bool saveExternalData(std::unique_ptr<tars::NetT>& netT,
                      const std::string& extraFileName);
void castParamsToHalf(std::unique_ptr<tars::NetT>& netT);
void AlignDenormalizedValue(std::unique_ptr<tars::NetT>& netT);
void addSparseInfo(std::unique_ptr<tars::NetT>& netT,
                   tars::Compression::Pipeline proto);
void fullQuantAndCoding(std::unique_ptr<tars::NetT>& netT,
                        tars::Compression::Pipeline proto);
void weightQuantAndCoding(std::unique_ptr<tars::NetT>& netT,
                          const modelConfig& config);
void addUUID(std::unique_ptr<tars::NetT>& netT,
             tars::Compression::Pipeline proto);

#endif  // COMMMON_UTILS_HPP

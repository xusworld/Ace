//
//  PostConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <ace/MNNDefine.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <sstream>

#include "ace_generated.h"
#include "config.hpp"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"

/**
 *@brief optimize MNN net
 */
MNN_PUBLIC std::unique_ptr<ace::NetT> optimizeNet(
    std::unique_ptr<ace::NetT>& netT, bool forTraining, modelConfig& config);

#endif  // OPTIMIZER_HPP

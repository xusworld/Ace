//
//  GemmCommon.hpp
//  MNN
//
//  Created by MNN on 2020/09/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GemmCommon_hpp
#define GemmCommon_hpp
#include <ace/MNNDefine.h>
#include <stdint.h>

void AVX2GemmPostTreat(float* C, size_t eSize, const size_t* parameter,
                       const float* postParameters, const float* bias);
#endif

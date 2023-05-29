//
//  InitNet.hpp
//  MNN
//
//  Created by MNN on 2018/09/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef CORE_INITNET_HPP
#define CORE_INITNET_HPP
#include "MNN_generated.h"
#include "core/Schedule.hpp"
#include "core/TensorUtils.hpp"

namespace tars {
MNN_PUBLIC bool needComputeOp(const Op* op);
MNN_PUBLIC bool initConstTensors(std::vector<std::shared_ptr<Tensor>>& tensors,
                                 const Net* net, Device* defaultBackend,
                                 Status& code);
// init Tensors by net
MNN_PUBLIC bool initTensors(std::vector<std::shared_ptr<Tensor>>& allTensors,
                            const Net* net);
// init Pipeline Infos by oplist and tensors
MNN_PUBLIC void initPipelineInfosFromOps(
    std::vector<Schedule::OpCacheInfo>& infos, std::vector<const Op*>& ops,
    const std::vector<std::shared_ptr<Tensor>>& allTensors);
// set input and output for allTensors by ops info
void setInputOutputForOps(std::vector<std::shared_ptr<Tensor>>& allTensors,
                          const std::vector<const Op*>& ops,
                          bool isStatic = false);
// init Pipeline Infos by net and tensors, set input and output info
void initPipelineInfosFromNet(std::vector<Schedule::OpCacheInfo>& infos,
                              const Net* net,
                              std::vector<std::shared_ptr<Tensor>>& allTensors);
}  // namespace tars

#endif

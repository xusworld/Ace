#pragma once

#include <stdio.h>

#include "ace_generated.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MNN_MEMORY_ALIGN_DEFAULT 64

/**
 * @brief alloc memory with given size & alignment.
 * @param size  given size. size should > 0.
 * @param align given alignment.
 * @return memory pointer.
 * @warning use `MNNMemoryFreeAlign` to free returned pointer.
 * @sa MNNMemoryFreeAlign
 */
void* MNNMemoryAllocAlign(size_t size, size_t align);

/**
 * @brief alloc memory with given size & alignment, and fill memory space with
 * 0.
 * @param size  given size. size should > 0.
 * @param align given alignment.
 * @return memory pointer.
 * @warning use `MNNMemoryFreeAlign` to free returned pointer.
 * @sa MNNMemoryFreeAlign
 */
void* MNNMemoryCallocAlign(size_t size, size_t align);

/**
 * @brief free aligned memory pointer.
 * @param mem   aligned memory pointer.
 * @warning do NOT pass any pointer NOT returned by `MNNMemoryAllocAlign` or
 * `MNNMemoryCallocAlign`.
 * @sa MNNMemoryAllocAlign
 * @sa MNNMemoryCallocAlign
 */
void MNNMemoryFreeAlign(void* mem);

#ifdef __cplusplus
}
#endif

namespace ace {
// init Tensors by net
bool initTensors(std::vector<std::shared_ptr<Tensor>>& allTensors,
                 const Net* net);
// init Pipeline Infos by oplist and tensors
void initPipelineInfosFromOps(
    std::vector<Schedule::PipelineInfo>& infos, std::vector<const Op*>& ops,
    const std::vector<std::shared_ptr<Tensor>>& allTensors);
// set input and output for allTensors by ops info
void setInputOutputForOps(std::vector<std::shared_ptr<Tensor>>& allTensors,
                          const std::vector<const Op*>& ops,
                          bool isStatic = false);
// init Pipeline Infos by net and tensors, set input and output info
void initPipelineInfosFromNet(std::vector<Schedule::PipelineInfo>& infos,
                              const Net* net,
                              std::vector<std::shared_ptr<Tensor>>& allTensors);
}  // namespace ace

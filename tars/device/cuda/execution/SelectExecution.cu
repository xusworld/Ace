//
//  SelectExecution.cpp
//  MNN
//
//  Created by MNN on 2021/12/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Selectoperation.h"
#include "core/Macro.h"
#include <cuda_runtime.h>

namespace tars {
namespace cuda {
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template<typename T>
__global__ void SELECT(const int size, const int* input0, const T* input1, const T* input2, T* output) {
    CUDA_KERNEL_LOOP(i, size) {
        if (input0[i] > 0) {
            output[i] = input1[i];
        } else {
            output[i] = input2[i];
        }
    }
}

SelectExecution::SelectExecution(Device* backend) : Operation(backend) {
    // Do nothing
}
Status SelectExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Do nothing
    return Status::OK();
}

Status SelectExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start SelectExecution onExecute...");
#endif
    auto runtime = static_cast<CUDADevice*>(backend())->getCUDARuntime();
    auto count = CUDABackend::realSize(inputs[0]);
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    if (static_cast<CUDADevice*>(backend())->useFp16()) {
        SELECT<<<block_num, threads_num>>>(count, (const int*)(inputs[0]->deviceId()), (const half*)(inputs[1]->deviceId()), (const half*)(inputs[2]->deviceId()), (half*)outputs[0]->deviceId());
    } else {
        SELECT<<<block_num, threads_num>>>(count, (const int*)(inputs[0]->deviceId()), (const float*)(inputs[1]->deviceId()), (const float*)(inputs[2]->deviceId()), (float*)outputs[0]->deviceId());
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end SelectExecution onExecute...");
#endif
    return Status::OK();
}


class SelectCreator : public CUDABackend::Creator {
public:
    virtual Operation* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const tars::Op* op, Device* backend) const override {
        return new SelectExecution(backend);
    }
};

CUDACreatorRegister<SelectCreator> __SelectExecution(OpType_Select);
} // namespace cuda
} // namespace tars

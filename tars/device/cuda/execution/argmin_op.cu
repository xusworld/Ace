//
//  ArgMinExecution.cpp
//  MNN
//
//  Created by MNN on 2022/06/29.
//  Copyright © 2018 - 2022, Alibaba Group Holding Limited
//
#include "ArgMinoperation.h"
#include "core/TensorUtils.hpp"
#include <MNN/AutoTime.hpp>

namespace tars {
namespace cuda {

template <typename T>
__global__ void ARGMIN(const int count, const int outside, const int inside, const int dim,
                         const T *input, int *output) {

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        const int o = i / inside;
        const int n = i % inside;

        int* outPtr = output + inside * o;
        const T* inpPtr = input + inside * dim * o;
        int index = 0;
        T minValue = inpPtr[n + 0 * inside];
        for(int j=1; j<dim; j++) {
            T value = inpPtr[n + j * inside];
            if(minValue > value) {
                index = j;
                minValue = value;
            }
        }
        outPtr[n] = index;
    }
    return;
}

ArgMinOp::ArgMinOp(const Op* op, Device *backend) : Operation(backend) {
    mOp = op;
    mAxis = mOp->main_as_ArgMax()->axis();
}

ArgMinOp::~ArgMinOp(){
    // Do nothing
}

Status ArgMinOp::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output  = outputs[0];

    if (mAxis < 0) {
        mAxis = input->dimensions() + mAxis;
    }

    mInside = 1;
    mOutside = 1;
    for (int i=0; i<mAxis; ++i) {
        mOutside *= input->length(i);
    }
    for (int i=mAxis+1; i<input->dimensions(); ++i) {
        mInside *= input->length(i);
    }
    mDim = input->length(mAxis);

    return Status::OK();
}

Status ArgMinOp::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend *>(backend())->getCUDARuntime();

    auto input = (void *)inputs[0]->deviceId();
    auto output = (void *)outputs[0]->deviceId();
    int count = mOutside * mInside;
    int block_num = runtime->blocks_num(count);
    int thread_num = runtime->threads_num();

    auto bytes = static_cast<CUDADevice*>(backend())->getBytes(inputs[0]);

    if(bytes == 4) {
        ARGMIN<<<block_num, thread_num>>>(count, mOutside, mInside, mDim, (const float*)input,(int *)output);
        checkKernelErrors;
    } else {
        ARGMIN<<<block_num, thread_num>>>(count, mOutside, mInside, mDim, (const half*)input,(int *)output);
        checkKernelErrors;
    }
    return Status::OK();
}

}
}

//
//  UnaryExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Unaryoperation.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "Raster.cuh"
#include "device/cuda/core/CUDABackend.hpp"
#include <cuda_runtime.h>

namespace tars {
namespace cuda {

void callUnary(void *input, void *output, size_t count, tars::CUDARuntime* runtime, halide_type_t data_type,
   tars::UnaryOpOperation op_type)
{
    Tensor::InsideDescribe::Region reg;
    reg.size[2] = count;
    UnaryBlit((uint8_t*)output, (const uint8_t*)input, reg.size, reg.src.stride, reg.dst.stride, data_type.bytes(), runtime, op_type);
    return;
}

UnaryExecution::UnaryExecution(UnaryOpOperation opType, Device* backend) : Operation(backend) {
    auto cudaBackend = static_cast<CUDADevice*>(backend);
    mRuntime      = cudaBackend->getCUDARuntime();
    mOpType = opType;
}
Status UnaryExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto shape = inputs[0]->shape();
    mCount = CUDABackend::realSize(inputs[0]);
    return Status::OK();
}

Status UnaryExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start UnaryExecution onExecute...");
#endif
    auto type = inputs[0]->getType();
    if (static_cast<CUDADevice*>(backend())->useFp16()) {
        type.bits = 16;
    }
    //MNN_PRINT("unary size:%d\n", mCount);
    callUnary((void*)inputs[0]->deviceId(), (void*)outputs[0]->deviceId(), mCount, mRuntime, type, mOpType);
#ifdef LOG_VERBOSE
    MNN_PRINT("end UnaryExecution onExecute...");
#endif
    return Status::OK();
}


__global__ void RELU(const float *input, float *output, size_t count, float slope) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float x = input[i];
    float y = x > 0 ? x : x * slope;
    output[i] = y;
  }
  return;
}

__global__ void RELU_Half(const half *input, half *output, size_t count, float slope) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float x = input[i];
    float y = x > 0 ? x : x * slope;
    output[i] = (half)y;
  }
  return;
}

class ReluExecution : public Operation {
public:
    ReluExecution(Device* bn, float slope) : Operation(bn) {
        mSlope = slope;
    }
    virtual ~ReluExecution() = default;
    Status onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto runtime = static_cast<CUDADevice*>(backend())->getCUDARuntime();
        auto count = CUDABackend::realSize(inputs[0]);
        int block_num = runtime->blocks_num(count);
        int threads_num = runtime->threads_num();
        auto input = inputs[0]->deviceId();
        auto output = outputs[0]->deviceId();
        if (static_cast<CUDADevice*>(backend())->useFp16()) {
            RELU_Half<<<block_num, threads_num>>>((half*)input, (half*)output, count, mSlope);
        } else {
            RELU<<<block_num, threads_num>>>((float*)input, (float*)output, count, mSlope);
        }
        return Status::OK();
    }
private:
    float mSlope;
};


template<typename T>
__global__ void CLAMP(const T *input, T *output, size_t count, float minV, float maxV) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float x = input[i];
    float y = min(max(x, minV), maxV);
    output[i] = y;
  }
  return;
}
class Relu6Execution : public Operation {
public:
    Relu6Execution(Device* bn, float minV, float maxV) : Operation(bn) {
        mMinV = minV;
        mMaxV = maxV;
    }
    virtual ~Relu6Execution() = default;
    Status onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto runtime = static_cast<CUDADevice*>(backend())->getCUDARuntime();
        auto count = CUDABackend::realSize(inputs[0]);
        int block_num = runtime->blocks_num(count);
        int threads_num = runtime->threads_num();
        auto input = inputs[0]->deviceId();
        auto output = outputs[0]->deviceId();
        if (static_cast<CUDADevice*>(backend())->useFp16()) {
            CLAMP<<<block_num, threads_num>>>((half*)input, (half*)output, count, mMinV, mMaxV);
        } else {
            CLAMP<<<block_num, threads_num>>>((float*)input, (float*)output, count, mMinV, mMaxV);
        }
        return Status::OK();
    }
private:
    float mMinV;
    float mMaxV;
};

template <typename T1, typename T2>
__global__ void CAST(T1 *input, T2 *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = (T2)(input[i]);
  }
  return;
}

template <typename T1, typename T2>
__global__ void CASTMIDFLOAT(T1 *input, T2 *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = (T2)((float)input[i]);
  }
  return;
}

__global__ void CASTBOOL(int32_t *input, int32_t *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i] > 0 ? 1 : 0;
  }
  return;
}

static DataType _mapDataType(DataType src) {
    if (DataType_DT_BOOL == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_INT64 == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_DOUBLE == src) {
        return DataType_DT_FLOAT;
    }
    return src;
}
class CastExecution : public Operation {
public:
    CastExecution(Device* bn, DataType dstType) : Operation(bn) {
        mDst = dstType;
    }
    virtual ~CastExecution() = default;
    Status onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto runtime = static_cast<CUDADevice*>(backend())->getCUDARuntime();
        auto count = CUDABackend::realSize(inputs[0]);
        int block_num = runtime->blocks_num(count);
        int threads_num = runtime->threads_num();
        auto input = inputs[0]->deviceId();
        auto output = outputs[0]->deviceId();
        auto dstT = _mapDataType(mDst);

        const auto &inputDataType = inputs[0]->getType();
        if (inputDataType.bytes() == 4 && mDst == tars::DataType_DT_BOOL) {
            CASTBOOL<<<block_num, threads_num>>>((int32_t*)input, (int32_t*)output, count);
            return Status::OK();
        }
        if (inputs[0]->buffer().type == outputs[0]->buffer().type) {
            runtime->memcpy((void*)output, (void*)input, count * static_cast<CUDADevice*>(backend())->getBytes(inputs[0]), MNNMemcpyDeviceToDevice, true);
            return Status::OK();
        }
        if (dstT == tars::DataType_DT_INT32 && halide_type_of<int8_t>() == inputDataType) {
            CAST<<<block_num, threads_num>>>((int8_t*)input, (int32_t*)output, count);
            return Status::OK();
        } else if (dstT == tars::DataType_DT_UINT8 && halide_type_of<int32_t>() == inputDataType) {
            CAST<<<block_num, threads_num>>>((int32_t*)input, (uint8_t*)output, count);
            return Status::OK();
        } else if (dstT == tars::DataType_DT_INT32 && halide_type_of<uint8_t>() == inputDataType) {
            CAST<<<block_num, threads_num>>>((uint8_t*)input, (int32_t*)output, count);
            return Status::OK();
        }
        if (static_cast<CUDADevice*>(backend())->useFp16()) {
            if (dstT == tars::DataType_DT_INT32 && halide_type_of<float>() == inputDataType) {
                CASTMIDFLOAT<<<block_num, threads_num>>>((half*)input, (int*)output, count);
            } else if (dstT == tars::DataType_DT_FLOAT && halide_type_of<int32_t>() == inputDataType) {
                CASTMIDFLOAT<<<block_num, threads_num>>>((int*)input, (half*)output, count);
            } else if (dstT == tars::DataType_DT_FLOAT && halide_type_of<uint8_t>() == inputDataType) {
                CASTMIDFLOAT<<<block_num, threads_num>>>((uint8_t*)input, (half*)output, count);
            } else if (dstT == tars::DataType_DT_FLOAT && halide_type_of<int8_t>() == inputDataType) {
                CASTMIDFLOAT<<<block_num, threads_num>>>((int8_t*)input, (half*)output, count);
            } else if (dstT == tars::DataType_DT_INT8 && halide_type_of<float>() == inputDataType) {
                CASTMIDFLOAT<<<block_num, threads_num>>>((half*)input, (int8_t*)output, count);
            } else if (dstT == tars::DataType_DT_UINT8 && halide_type_of<float>() == inputDataType) {
                CASTMIDFLOAT<<<block_num, threads_num>>>((half*)input, (uint8_t*)output, count);
            }
        } else {
            if (dstT == tars::DataType_DT_INT32 && halide_type_of<float>() == inputDataType) {
                CASTMIDFLOAT<<<block_num, threads_num>>>((float*)input, (int*)output, count);
            } else if (dstT == tars::DataType_DT_FLOAT && halide_type_of<int32_t>() == inputDataType) {
                CASTMIDFLOAT<<<block_num, threads_num>>>((int*)input, (float*)output, count);
            } else if (dstT == tars::DataType_DT_FLOAT && halide_type_of<uint8_t>() == inputDataType) {
                CASTMIDFLOAT<<<block_num, threads_num>>>((uint8_t*)input, (float*)output, count);
            } else if (dstT == tars::DataType_DT_FLOAT && halide_type_of<int8_t>() == inputDataType) {
                CASTMIDFLOAT<<<block_num, threads_num>>>((int8_t*)input, (float*)output, count);
            } else if (dstT == tars::DataType_DT_INT8 && halide_type_of<float>() == inputDataType) {
                CASTMIDFLOAT<<<block_num, threads_num>>>((float*)input, (int8_t*)output, count);
            } else if (dstT == tars::DataType_DT_UINT8 && halide_type_of<float>() == inputDataType) {
                CASTMIDFLOAT<<<block_num, threads_num>>>((float*)input, (uint8_t*)output, count);
            }
        }
        return Status::OK();
    }
private:
    DataType mDst;
};


class UnaryCreator : public CUDABackend::Creator {
public:
    virtual Operation* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const tars::Op* op, Device* backend) const override {
        if (op->type() == OpType_UnaryOp) {
            return new UnaryExecution(op->main_as_UnaryOp()->opType(), backend);
        }
        if (op->type() == OpType_Sigmoid) {
            return new UnaryExecution(UnaryOpOperation_SIGMOID, backend);
        }
        if (op->type() == OpType_TanH) {
            return new UnaryExecution(UnaryOpOperation_TANH, backend);
        }
        if (op->type() == OpType_ReLU) {
            float slope = 0.0f;
            if (nullptr != op->main_as_Relu()) {
                slope = op->main_as_Relu()->slope();
            }
            return new ReluExecution(backend, slope);
        }
        if (op->type() == OpType_ReLU6) {
            float minV = 0.0f;
            float maxV = 6.0f;
            if (nullptr != op->main()) {
                auto p = op->main_as_Relu6();
                minV = p->minValue();
                maxV = p->maxValue();
            }
            return new Relu6Execution(backend, minV, maxV);
        }
        if (op->type() == OpType_Cast) {
            return new CastExecution(backend, op->main_as_CastParam()->dstT());
        }
        return nullptr;
    }
};

CUDACreatorRegister<UnaryCreator> __UnaryExecution(OpType_UnaryOp);
CUDACreatorRegister<UnaryCreator> __SigmoidExecution(OpType_Sigmoid);
CUDACreatorRegister<UnaryCreator> __TanhExecution(OpType_TanH);
CUDACreatorRegister<UnaryCreator> __ReluExecution(OpType_ReLU);
CUDACreatorRegister<UnaryCreator> __Relu6Execution(OpType_ReLU6);
CUDACreatorRegister<UnaryCreator> __CastExecution(OpType_Cast);
} // namespace cuda
} // namespace tars

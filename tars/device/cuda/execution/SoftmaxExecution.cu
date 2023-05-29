#include "Softmaxoperation.h"
#include "core/TensorUtils.hpp"
namespace tars {
namespace cuda {

template <typename T>
__global__ void SOFTMAX(const T *input, T *output,
    const int inside,
    const int axis,
    const int outside,
    const int count
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / inside;
        int x = i % inside;
        const T* src = input + y * axis * inside + x;
        T* dst = output + y * axis * inside + x;
        float maxValue = (float)src[0];
        for (int z=1; z<axis; ++z) {
            maxValue = max(maxValue, src[z * inside]);
        }
        float sumValue = 0.0;
        for (int z=0; z<axis; ++z) {
            sumValue = sumValue + exp((float)src[z * inside] - maxValue);
        }
        sumValue = 1.0 / sumValue;
        for (int z=0; z<axis; ++z) {
            dst[z*inside] = (T)(exp((float)src[z * inside] - maxValue) * sumValue);
        }
    }
}

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T warpReduceMax(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
  return val;
}

template <typename T>
__global__ void SOFTMAX_WARP_32(const T *input, T *output,
    const int inside,
    const int axis,
    const int outside,
    const int count
) {
    int idx_outside = blockIdx.x / inside;
    int idx_inside = blockIdx.x -  idx_outside * inside;

    auto src = input + idx_outside * axis * inside + idx_inside;
    float local_src = -FLT_MAX;
    __shared__ float maxValue;
    __shared__ float sumValue;
    int tid = threadIdx.x;
    if(tid < axis) {
        local_src = (float)(src[tid * inside]);
    }
    float maxRes = warpReduceMax<float>(local_src);
    if(tid == 0)
        maxValue = maxRes;
    __syncthreads();


    float local_exp = 0.0f;
    if(tid < axis) {
        local_exp = exp(local_src - maxValue);
    }

    float sumRes = warpReduceSum<float>(local_exp);
    if(tid == 0)
        sumValue = sumRes;
    __syncthreads();

    sumValue = 1.0 / sumValue;

    if(tid < axis) {
        output[(idx_outside * axis + tid) * inside + idx_inside] = (T)(local_exp * sumValue);
    }
}

SoftmaxExecution::SoftmaxExecution(int axis, Device *backend) : Operation(backend) {
    mAxis = axis;
}

SoftmaxExecution::~SoftmaxExecution() {
    //
}

Status SoftmaxExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input           = inputs[0];
    const int dimensions = input->buffer().dimensions;
    auto runtime = static_cast<CUDADevice*>(backend())->getCUDARuntime();
    int axis = mAxis;
    if (axis < 0) {
        axis += dimensions;
    }

    const auto layout = TensorUtils::getDescribe(input)->dimensionFormat;
    mNeedUnpackC4     = layout == MNN_DATA_FORMAT_NC4HW4;
    if (mNeedUnpackC4) {    
        TensorUtils::copyShape(input, &mStorage);
        TensorUtils::getDescribe(&mStorage)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        mStorage.buffer().dimensions    = dimensions;
        mStorage.buffer().type          = input->getType();
        backend()->onAcquireBuffer(&mStorage, Device::DYNAMIC);
    }

    int inside = 1;
    int outside = 1;
    int dims   = input->buffer().dimensions;
    for (int i = 0; i < axis; ++i) {
        outside *= input->length(i);
    }
    for (int i = axis + 1; i < dims; ++i) {
        inside *= input->length(i);
    }

    if (mNeedUnpackC4) {
        backend()->onReleaseBuffer(&mStorage, Device::DYNAMIC);
    }

    mCpuParam.inside = inside;
    mCpuParam.outside = outside;
    mCpuParam.axis = input->length(axis);

    // printf("\nsoftmax:%d-%d-%d, %d-%d\n", mCpuParam.inside, mCpuParam.outside, mCpuParam.axis, mNeedUnpackC4, axis);
    return Status::OK();
}

Status SoftmaxExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = (void*)inputs[0]->deviceId();
    auto output = (void*)outputs[0]->deviceId();
    auto dst = output;

    if (mNeedUnpackC4) {
        backend()->onCopyBuffer(inputs[0], &mStorage);
        input = (void*)mStorage.deviceId();
        dst = (void*)mStorage.deviceId();
    }

    //MNN_PRINT("softmax input dims:%d, size:%d-%d-%d-%d\n", inputs[0]->dimensions(), inputs[0]->batch(), inputs[0]->height(), inputs[0]->width(), inputs[0]->channel());
    //MNN_PRINT("softmax storage dims:%d, size:%d-%d-%d-%d\n", mStorage.dimensions(), mStorage.batch(), mStorage.height(), mStorage.width(), mStorage.channel());

    auto runtime = static_cast<CUDADevice*>(backend())->getCUDARuntime();
    int inside = mCpuParam.inside;
    int outside = mCpuParam.outside;
    int axis = mCpuParam.axis;
    int count = inside * outside;
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    if (static_cast<CUDADevice*>(backend())->useFp16()) {
        if(axis <= 32) {
            threads_num = 32;
            block_num = count;
            SOFTMAX_WARP_32<<<block_num, threads_num>>>((const half*)input, (half*)dst, inside, axis, outside, count);
        } else {
            SOFTMAX<<<block_num, threads_num>>>((const half*)input, (half*)dst, inside, axis, outside, count);
        }
    } else {
        if(axis <= 32) {
            threads_num = 32;
            block_num = count;
            SOFTMAX_WARP_32<<<block_num, threads_num>>>((const float*)input, (float*)dst, inside, axis, outside, count);
        } else {
            SOFTMAX<<<block_num, threads_num>>>((const float*)input, (float*)dst, inside, axis, outside, count);
        }
    }
    if (mNeedUnpackC4) {
        backend()->onCopyBuffer(&mStorage, outputs[0]);
    }

    return Status::OK();
}

class SoftmaxCreator : public CUDABackend::Creator {
public:
    virtual Operation* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const tars::Op* op, Device* backend) const override {
        auto type = inputs[0]->getType();
        if (type.code != halide_type_float) {
            MNN_PRINT("softmax data type:%s not support", type.code);
            return nullptr;
        }
        auto axis = op->main_as_Axis()->axis();
        return new SoftmaxExecution(axis, backend);
    }
};

static CUDACreatorRegister<SoftmaxCreator> __init(OpType_Softmax);
}
}

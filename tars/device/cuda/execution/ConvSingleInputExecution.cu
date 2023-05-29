//
//  ConvSingleInputExecution.cpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvSingleInputoperation.h"
#include "ConvWinogradoperation.h"
#include "ConvCutlassoperation.h"
#include "int8/ConvInt8Cutlassoperation.h"
#include "device/cuda/core/CUDATools.hpp"

namespace tars {
namespace cuda {

class CUDAConvolutionCreator : public CUDABackend::Creator {
public:
    virtual Operation* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
            const tars::Op* op, Device* backend) const override {
        if (nullptr != op->main_as_Convolution2D()->quanParameter()) {
            auto quan = op->main_as_Convolution2D()->quanParameter();
            if (1 == quan->type() || 2 == quan->type()) {
                if (quan->has_scaleInt()) {
                    // Don't support IDST-int8 because of error
                    return nullptr;
                }
            }
        }

#ifdef USE_MNN_CONV

        std::shared_ptr<ConvSingleInputExecution::Resource> resource(new ConvSingleInputExecution::Resource(backend, op));
        return new ConvSingleInputExecution(backend, op, resource);

#else
        auto conv = op->main_as_Convolution2D()->common();
        if(ConvWinogradExecution::isValid(op->main_as_Convolution2D())) { // inputs[0] is invalid now.
            //printf("%dx%ds%dd%d\n", conv->kernelX(), conv->kernelY(), conv->strideX(), conv->dilateX());

            std::shared_ptr<ConvWinogradExecution::Resource> resource(new ConvWinogradExecution::Resource(backend, op));
            return new ConvWinogradExecution(backend, op, resource);
        }

        std::shared_ptr<ConvCutlassExecution::Resource> resource(new ConvCutlassExecution::Resource(backend, op));
        return new ConvCutlassExecution(backend, op, resource);
#endif

    }
};


class CUDAConvolutionInt8Creator : public CUDABackend::Creator {
public:
    virtual Operation* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
            const tars::Op* op, Device* backend) const override {
        std::shared_ptr<ConvInt8CutlassExecution::Resource> resource(new ConvInt8CutlassExecution::Resource(backend, op));
        return new ConvInt8CutlassExecution(backend, op, resource);
    }
};

CUDACreatorRegister<CUDAConvolutionCreator> __ConvExecution(OpType_Convolution);
CUDACreatorRegister<CUDAConvolutionInt8Creator> __ConvInt8Execution(OpType_ConvInt8);

}// namespace cuda
}// namespace tars

//
//  WrapExecution.cpp
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/TensorUtils.hpp"
#include "core/WrapExecution.hpp"
namespace ace {
bool WrapExecution::needWrap(const Tensor* input, Backend* curBackend) {
  auto des = TensorUtils::getDescribe(input);
  auto bn = des->backend;
  DeviceType type = DeviceType::X86;
  int pack = 4;
  int bytes = 4;
  if (nullptr != bn) {
    type = bn->type();
    if (type == DeviceType::X86_EXTENSION) {
      auto core = static_cast<CPUBackend*>(bn)->functions();
      pack = core->pack;
      bytes = core->bytes;
    }
  }
  if (type == curBackend->type()) {
    return false;
    ;
  }
  bool srcCpu = (type == DeviceType::X86_EXTENSION || type == DeviceType::X86);
  bool dstCpu = ((curBackend->type() == DeviceType::X86_EXTENSION) ||
                 (curBackend->type() == DeviceType::X86));
  if (srcCpu && dstCpu) {
    auto dstCore = static_cast<CPUBackend*>(curBackend)->functions();
    if (dstCore->bytes == bytes) {
      if (dstCore->pack == pack || des->dimensionFormat != DATA_FORMAT_NC4HW4) {
        return false;
      }
    }
  }
  return true;
}

WrapExecution::WrapExecution(Backend* CPUBackend,
                             std::shared_ptr<Execution> execution,
                             bool isStatic)
    : Execution(execution->backend()),
      mCPUBackend(CPUBackend),
      mExecution(execution) {
  mValid = execution->valid();
  mStatic = isStatic;
}

Tensor* WrapExecution::_getCopyTensor(Tensor* inputTensor) {
  auto dstBackend = mExecution->backend();
  auto inputDes = TensorUtils::getDescribe(inputTensor);
  auto srcBackend = inputDes->backend;
  if (nullptr == srcBackend) {
    srcBackend = mCPUBackend;
  }
  // CPU -> CPU or XPU -> XPU
  // if (srcBackend == dstBackend) {
  if (srcBackend->type() == dstBackend->type()) {
    return inputTensor;
  }
  auto iter = mInputMaps.find(inputTensor);
  if (iter != mInputMaps.end()) {
    return std::get<2>(iter->second).get();
  }
  // CPU -> XPU
  if (srcBackend->type() == mCPUBackend->type()) {
    std::shared_ptr<Tensor> wrapTensor(new Tensor);
    TensorUtils::copyShape(inputTensor, wrapTensor.get(), true);
    TensorUtils::adjustTensorForCompability(wrapTensor.get());
    wrapTensor->buffer().type = inputTensor->buffer().type;
    TensorUtils::getDescribe(wrapTensor.get())->quantAttr =
        TensorUtils::getDescribe(inputTensor)->quantAttr;
    mInputMaps.insert(std::make_pair(
        inputTensor, std::make_tuple(dstBackend, dstBackend, wrapTensor)));
    return wrapTensor.get();
  }
  // XPU -> CPU
  if (dstBackend->type() == mCPUBackend->type()) {
    std::shared_ptr<Tensor> wrapTensor(new Tensor);
    TensorUtils::copyShape(inputTensor, wrapTensor.get(), true);
    wrapTensor->buffer().type = inputTensor->buffer().type;
    TensorUtils::adjustTensorForCompability(wrapTensor.get());
    TensorUtils::getDescribe(wrapTensor.get())->quantAttr =
        TensorUtils::getDescribe(inputTensor)->quantAttr;
    mInputMaps.insert(std::make_pair(
        inputTensor, std::make_tuple(mCPUBackend, srcBackend, wrapTensor)));
    return wrapTensor.get();
  }
  // XPU -> CPU -> XPU'
  std::shared_ptr<Tensor> midTensor(new Tensor);
  std::shared_ptr<Tensor> wrapTensor(new Tensor);
  TensorUtils::copyShape(inputTensor, midTensor.get(), true);
  TensorUtils::copyShape(inputTensor, wrapTensor.get(), true);
  TensorUtils::adjustTensorForCompability(wrapTensor.get());
  TensorUtils::adjustTensorForCompability(midTensor.get());
  TensorUtils::getDescribe(midTensor.get())->usage =
      TensorUtils::getDescribe(inputTensor)->usage;
  TensorUtils::getDescribe(midTensor.get())->quantAttr =
      TensorUtils::getDescribe(inputTensor)->quantAttr;
  midTensor->buffer().type = inputTensor->buffer().type;
  wrapTensor->buffer().type = inputTensor->buffer().type;
  mInputMaps.insert(std::make_pair(
      inputTensor, std::make_tuple(mCPUBackend, srcBackend, midTensor)));
  mInputMaps.insert(std::make_pair(
      midTensor.get(), std::make_tuple(dstBackend, dstBackend, wrapTensor)));
  return wrapTensor.get();
}

ErrorCode WrapExecution::onResize(const std::vector<Tensor*>& inputs,
                                  const std::vector<Tensor*>& outputs) {
  mWrapInputTensors.resize(inputs.size());
  mInputMaps.clear();

  auto dstBackend = mExecution->backend();
  for (int i = 0; i < inputs.size(); ++i) {
    auto inputTensor = inputs[i];
    auto des = TensorUtils::getDescribe(inputTensor);
    if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
      MNN_ASSERT(inputs.size() == 1);
      mWrapForRaster.reset(new Tensor);
      TensorUtils::copyShape(inputTensor, mWrapForRaster.get(), true);
      mWrapForRaster->buffer().type = inputTensor->buffer().type;
      auto wrapDes = TensorUtils::getDescribe(mWrapForRaster.get());
      wrapDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
      wrapDes->regions = des->regions;
      for (auto& r : wrapDes->regions) {
        r.origin = _getCopyTensor(r.origin);
      }
      mWrapInputTensors[i] = mWrapForRaster.get();
    } else {
      mWrapInputTensors[i] = _getCopyTensor(inputTensor);
    }
  }

  for (int i = 0; i < outputs.size(); ++i) {
    MNN_ASSERT(TensorUtils::getDescribe(outputs[i])->backend == dstBackend);
  }
  bool memoryAllocSuccess = true;
  // acquire memory, copy const tensors
  for (auto& iter : mInputMaps) {
    auto backend = std::get<0>(iter.second);
    auto converter = std::get<1>(iter.second);
    auto src = iter.first;
    auto dst = std::get<2>(iter.second).get();

    if (TensorUtils::getDescribe(src)->usage == TensorUsage::CONSTANT &&
        mStatic) {
      memoryAllocSuccess =
          backend->onAcquireBuffer(dst, Backend::DYNAMIC_SEPERATE);
      if (memoryAllocSuccess) {
        converter->onCopyBuffer(src, dst);
        TensorUtils::getDescribe(dst)->usage =
            TensorUtils::getDescribe(src)->usage;
      }
    } else {
      memoryAllocSuccess = backend->onAcquireBuffer(dst, Backend::DYNAMIC);
    }
  }
  if (!memoryAllocSuccess) {
    return OUT_OF_MEMORY;
  }

  // do resize
  auto result = mExecution->onResize(mWrapInputTensors, outputs);

  // release memory
  for (auto& iter : mInputMaps) {
    auto backend = std::get<0>(iter.second);
    auto dst = std::get<2>(iter.second).get();

    if (TensorUtils::getDescribe(dst)->usage == TensorUsage::CONSTANT &&
        mStatic) {
      backend->onReleaseBuffer(dst, Backend::DYNAMIC_SEPERATE);
    } else {
      backend->onReleaseBuffer(dst, Backend::DYNAMIC);
    }
  }
  return result;
}

ErrorCode WrapExecution::onExecute(const std::vector<Tensor*>& inputs,
                                   const std::vector<Tensor*>& outputs) {
  MNN_ASSERT(mWrapInputTensors.size() == inputs.size());

  // copy variant tensors
  for (auto& iter : mInputMaps) {
    auto converter = std::get<1>(iter.second);
    auto src = iter.first;
    auto dst = std::get<2>(iter.second).get();
    if (TensorUtils::getDescribe(src)->usage != TensorUsage::CONSTANT ||
        (!mStatic)) {
      converter->onCopyBuffer(src, dst);
    }
  }
  auto code = mExecution->onExecute(mWrapInputTensors, outputs);
  return code;
}

}  // namespace ace

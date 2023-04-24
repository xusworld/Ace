//
//  BF16Backend.cpp
//  MNN
//
//  Created by MNN on 2020/01/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>

#include "BF16Backend.hpp"
#include "BF16Functions.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"
#include "core/BufferAllocator.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/TensorUtils.hpp"
namespace ace {

void registerBF16Ops();
static std::map<OpType, BF16Backend::BF16Creator*>* gInstance = nullptr;
// The Function Will be Called in init
extern void registerBF16Backend() {
  gInstance = new std::map<OpType, BF16Backend::BF16Creator*>;
  bool success = BF16Functions::init();
  if (success) {
    registerBF16Ops();
  }
}
bool BF16Backend::addBF16Creator(OpType t, BF16Creator* ct) {
  auto creatorContainer = gInstance;
  if (creatorContainer->find(t) == creatorContainer->end()) {
    creatorContainer->insert(std::make_pair(t, ct));
  }
  return true;
}

BF16Backend::BF16Backend(const CPURuntime* runtime)
    : CPUBackend(runtime, BackendConfig::Precision_Low,
                 DeviceType::X86_EXTENSION) {
  mCoreFunctions = BF16Functions::get();
}

BF16Backend::~BF16Backend() {
  // nothing to do
}

Execution* BF16Backend::onCreate(const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs,
                                 const ace::Op* op) {
  for (auto t : outputs) {
    if (t->getType().code != halide_type_float) {
      return nullptr;
    }
  }
  auto quantInfo = OpCommonUtils::getQuantInfo(inputs);
  if (quantInfo.first) {
    return nullptr;
  }
  bool originCreate = OpCommonUtils::opCompabilityForLowp(op);
  if (originCreate) {
    return CPUBackend::onCreate(inputs, outputs, op);
  }
  auto creatorContainer = gInstance;
  auto iter = creatorContainer->find(op->type());

  if (iter == creatorContainer->end()) {
    return nullptr;
  }
  auto exe = iter->second->onCreate(inputs, outputs, op, this);
  if (exe == nullptr) {
    return nullptr;
  }
  return exe;
}

static int _getAliginSize(const halide_buffer_t& buffer, DATA_FORMAT format) {
  // The default data type of input tensor for arm82 backend is FLOAT32.
  // However, BF16Backend default data type is FLOAT16, so check whether data
  // type is FLOAT32, then divide size by 2
  int size = sizeof(int16_t);
  const int dimensions = buffer.dimensions;
  for (int i = 0; i < dimensions; i++) {
    int currentDimSize = buffer.dim[i].extent;
    if (format == DATA_FORMAT_NC4HW4 && 1 == i) {
      currentDimSize = ALIGN_UP4(currentDimSize);
    }
    size *= currentDimSize;
  }
  return size;
}

bool BF16Backend::onAcquireBuffer(const Tensor* nativeTensor,
                                  StorageType storageType) {
  // arm82 backend tensor data type is fp16 default
  auto tensor = const_cast<Tensor*>(nativeTensor);
  auto& buffer = tensor->buffer();
  if (buffer.type != halide_type_of<float>()) {
    return CPUBackend::onAcquireBuffer(nativeTensor, storageType);
  }
  auto res = allocBuffer(
      _getAliginSize(buffer,
                     TensorUtils::getDescribe(nativeTensor)->dimensionFormat),
      (Tensor*)nativeTensor, storageType);
  if (!res) {
    return false;
  }
  // Set mask in device for easy to determine
  buffer.device = 1;
  return true;
}

void BF16Backend::onCopyBuffer(const Tensor* srcTensor,
                               const Tensor* dstTensor) const {
  auto& ib = srcTensor->buffer();
  auto& ob = dstTensor->buffer();
  if (ib.type.code != halide_type_float) {
    CPUBackend::onCopyBuffer(srcTensor, dstTensor);
    return;
  }
  auto source = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
  auto dest = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
  auto srcType = DeviceType::X86;
  if (ib.device != 0) {
    srcType = DeviceType::X86_EXTENSION;
  }
  auto dstType = DeviceType::X86;
  if (ob.device != 0) {
    dstType = DeviceType::X86_EXTENSION;
  }
  if (srcType == dstType) {
    ErrorCode code = ErrorCode::NO_ERROR;
    auto tup = CPUTensorConverter::splitDimensions(srcTensor->buffer(), source);
    int area = std::get<1>(tup), batch = std::get<0>(tup),
        channel = std::get<2>(tup);
    if (srcType == DeviceType::X86) {
      code = CPUTensorConverter::convert(
          srcTensor->host<void>(), dstTensor->host<void>(), source, dest, batch,
          area, channel, 4, MNNGetCoreFunctions());
    } else {
      code = CPUTensorConverter::convert(
          srcTensor->host<void>(), dstTensor->host<void>(), source, dest, batch,
          area, channel, 2, mCoreFunctions);
    }
    MNN_ASSERT(code == ErrorCode::NO_ERROR);
    return;
  }
  // Use CPU Copy to turn save format
  std::shared_ptr<Tensor> tempTensor;
  if (source != dest) {
    if (srcType == DeviceType::X86) {
      tempTensor.reset(Tensor::create<float>(
          dstTensor->shape(), nullptr, TensorUtils::getDimType(dstTensor)));
      MNNCPUCopyBuffer(srcTensor, tempTensor.get());
      srcTensor = tempTensor.get();
      source = dest;
    } else {
      tempTensor.reset(
          Tensor::create<float>(srcTensor->shape(), nullptr,
                                TensorUtils::getDimType(srcTensor)),
          [dstTensor](void* ptr) {
            auto tempT = (Tensor*)ptr;
            MNNCPUCopyBuffer(tempT, dstTensor);
            delete tempT;
          });
      dstTensor = tempTensor.get();
      dest = source;
    }
  }
  // MNN_PRINT("%d, %d - %d, %d\n", source, srcType, dest, dstType);
  //  The format is the same, just convert fp32-fp16
  const int elemenSize = srcTensor->elementSize();
  // copy and quantize/dequantize data
  if (srcType == DeviceType::X86) {
    const auto src = srcTensor->host<float>();
    auto dst = dstTensor->host<int16_t>();
    BF16Functions::get()->MNNFp32ToLowp(src, dst, elemenSize);
    return;
  }
  if (srcType == DeviceType::X86_EXTENSION) {
    const auto src = srcTensor->host<int16_t>();
    auto dst = dstTensor->host<float>();
    BF16Functions::get()->MNNLowpToFp32(src, dst, elemenSize);
    return;
  }
  return;
}

}  // namespace ace

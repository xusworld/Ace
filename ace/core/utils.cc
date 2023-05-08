#include <glog/logging.h>
#include <stdint.h>
#include <stdlib.h>

#include <iostream>
#include <unordered_map>

#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "utils.h"

static inline void** alignPointer(void** ptr, size_t alignment) {
  return (void**)((intptr_t)((unsigned char*)ptr + alignment - 1) & -alignment);
}

extern "C" void* MNNMemoryAllocAlign(size_t size, size_t alignment) {
  MNN_ASSERT(size > 0);

#ifdef MNN_DEBUG_MEMORY
  return malloc(size);
#else
  void** origin = (void**)malloc(size + sizeof(void*) + alignment);
  MNN_ASSERT(origin != NULL);
  if (!origin) {
    return NULL;
  }

  void** aligned = alignPointer(origin + 1, alignment);
  aligned[-1] = origin;
  return aligned;
#endif
}

extern "C" void* MNNMemoryCallocAlign(size_t size, size_t alignment) {
  MNN_ASSERT(size > 0);

#ifdef MNN_DEBUG_MEMORY
  return calloc(size, 1);
#else
  void** origin = (void**)calloc(size + sizeof(void*) + alignment, 1);
  MNN_ASSERT(origin != NULL)
  if (!origin) {
    return NULL;
  }
  void** aligned = alignPointer(origin + 1, alignment);
  aligned[-1] = origin;
  return aligned;
#endif
}

extern "C" void MNNMemoryFreeAlign(void* aligned) {
#ifdef MNN_DEBUG_MEMORY
  free(aligned);
#else
  if (aligned) {
    void* origin = ((void**)aligned)[-1];
    free(origin);
  }
#endif
}

namespace ace {

// 初始化所有 tensors
bool initTensors(std::vector<std::shared_ptr<Tensor>>& tensors,
                 const Net* net) {
  auto describes = net->extraTensorDescribe();
  std::vector<const TensorDescribe*> des(tensors.size());
  if (describes) {
    for (int i = 0; i < describes->size(); i++) {
      int index = describes->GetAs<TensorDescribe>(i)->index();
      des[index] = describes->GetAs<TensorDescribe>(i);
    }
  }

  bool valid = true;
  for (int i = 0; i < tensors.size(); ++i) {
    tensors[i].reset(new Tensor(4));  // NCHW, TODO
    tensors[i]->setType(DataType_DT_FLOAT);
    if (des[i] != nullptr && des[i]->quantInfo()) {
      TensorUtils::getDescribe(tensors[i].get())
          ->quantAttr.reset(new QuantAttr);
      auto quant = TensorUtils::getDescribe(tensors[i].get())->quantAttr.get();
      quant->scale = des[i]->quantInfo()->scale();
      quant->zero = des[i]->quantInfo()->zero();
      quant->min = des[i]->quantInfo()->min();
      quant->max = des[i]->quantInfo()->max();
    }
  }

  // Set Input Tensor, if the type of input is not the same with
  // ExtraTensorDescribe, use input parameter
  for (int opIndex = 0; opIndex < net->oplists()->size(); ++opIndex) {
    auto op = net->oplists()->GetAs<Op>(opIndex);

    if (OpType_Input == op->type()) {
      CHECK(nullptr != op->outputIndexes());
      CHECK(op->outputIndexes()->size() == 1);

      LOG(INFO) << "name : " << op->name()->str();
      LOG(INFO) << "op output indexes : " << op->outputIndexes()->size();
      if (opIndex == 4) {
        exit(0);
      }
      auto index = op->outputIndexes()->data()[0];

      auto tensor = tensors[index].get();
      auto& tb = tensor->buffer();
      auto inputParam = op->main_as_Input();

      LOG(INFO) << "inputParam->dims(): " << inputParam->dims();
      if (auto idims = inputParam->dims()) {
        for (int i = 0; i < idims->size(); ++i) {
          int extent = idims->data()[i];
          // dim-0 is batch(when input batch is -1, set it to be 1, ignore other
          // dim)
          if (i == 0 && extent == -1) extent = 1;

          if (extent < 0) valid = false;

          tb.dim[i].extent = extent;
        }
        tb.dimensions = idims->size();
      } else {
        tb.dimensions = 0;
      }
      tensor->setType(inputParam->dtype());
      TensorUtils::getDescribe(tensor)->dimensionFormat = inputParam->dformat();
    }
  }
  return valid;
}

void initPipelineInfosFromOps(
    std::vector<Schedule::PipelineInfo>& infos, std::vector<const Op*>& ops,
    const std::vector<std::shared_ptr<Tensor>>& allTensors) {
  LOG(INFO) << "initPipelineInfosFromOps";

  for (const Op* op : ops) {
    Schedule::PipelineInfo opInfo;
    // 遍历所有Ops，保存每一个 op 的输入输出
    opInfo.op = op;
    if (nullptr != op->outputIndexes()) {
      LOG(INFO) << "op->outputIndexes()->size(): "
                << op->outputIndexes()->size();
      auto data = op->outputIndexes()->data();
      for (int j = 0; j < op->outputIndexes()->size(); ++j) {
        opInfo.outputs.push_back(allTensors[data[j]].get());
      }
    }
    if (nullptr != op->inputIndexes()) {
      LOG(INFO) << "op->inputIndexes()->size(): " << op->inputIndexes()->size();
      auto data = op->inputIndexes()->data();
      for (int j = 0; j < op->inputIndexes()->size(); ++j) {
        opInfo.inputs.push_back(allTensors[data[j]].get());
      }
    }
    if (op->type() != OpType_Input) {
      infos.emplace_back(std::move(opInfo));
    }
  }
}

// 设置输出输出
void setInputOutputForOps(std::vector<std::shared_ptr<Tensor>>& allTensors,
                          const std::vector<const Op*>& ops, bool isStatic) {
  LOG(INFO) << "setInputOutputForOps: isStatic = " << isStatic;
  std::set<int> inputIndexes;
  std::set<int> outputIndexes;
  // 0. deal virtual tensor for static model:
  // when : A (Any_Op) -----> B (Raster_Op)
  // the tensor will be like below:
  //      A_outputs : a_tensor
  //      B_inputs  : b_tensor (virtual)
  //      b_tensor.describe.origin = a_tensor_ptr
  // b_tensor is not a InputTensot, a_tensor is not a OutputTensor
  // so add b_tensor to OutputIndexes, a_tensor to InputIndexes.
  if (isStatic) {
    std::unordered_map<Tensor*, int> tensorMap;
    for (int index = 0; index < allTensors.size(); index++) {
      tensorMap.insert(std::make_pair(allTensors[index].get(), index));
    }
    for (int index = 0; index < allTensors.size(); index++) {
      auto des = TensorUtils::getDescribe(allTensors[index].get());
      for (int i = 0; i < des->regions.size(); i++) {
        outputIndexes.insert(index);
        MNN_ASSERT(tensorMap.find(des->regions[i].origin) != tensorMap.end());
        int x = tensorMap[des->regions[i].origin];
        inputIndexes.insert(x);
      }
    }
  }

  // 1. insert all output/input index in outputIndexes/inputIndexes
  for (auto op : ops) {
    if (nullptr != op->outputIndexes()) {
      auto data = op->outputIndexes()->data();
      for (int j = 0; j < op->outputIndexes()->size(); ++j) {
        outputIndexes.insert(data[j]);
      }
    }

    if (nullptr != op->inputIndexes()) {
      auto data = op->inputIndexes()->data();
      for (int j = 0; j < op->inputIndexes()->size(); ++j) {
        inputIndexes.insert(data[j]);
      }
    }

    MNN_ASSERT(OpType_Input != op->type());
  }

  LOG(INFO) << "outputIndexes.size(): " << outputIndexes.size();
  LOG(INFO) << "inputIndexes.size(): " << inputIndexes.size();

  // 2. the index in outputIndexes/inputIndexed but not in
  // inputIndexes/outputIndexes is output/input
  std::set<int> input;
  std::set<int> output;
  std::set_difference(outputIndexes.begin(), outputIndexes.end(),
                      inputIndexes.begin(), inputIndexes.end(),
                      std::inserter(output, output.begin()));
  std::set_difference(inputIndexes.begin(), inputIndexes.end(),
                      outputIndexes.begin(), outputIndexes.end(),
                      std::inserter(input, input.begin()));

  for (auto item : input) {
    LOG(INFO) << "input: " << item;
  }
  for (auto item : output) {
    LOG(INFO) << "output: " << item;
  }

  // 3. set usage for Tensor by index
  for (auto index : input) {
    if (TensorUtils::getDescribe(allTensors[index].get())->usage ==
        TensorUsage::CONSTANT) {
      continue;
    }
    // MNN_PRINT("%d - %p: input\n", index, allTensors[index].get());
    TensorUtils::getDescribe(allTensors[index].get())->usage =
        TensorUsage::INPUT;
  }
  for (auto index : output) {
    TensorUtils::getDescribe(allTensors[index].get())->usage =
        TensorUsage::OUTPUT;
  }
}

void initPipelineInfosFromNet(
    std::vector<Schedule::PipelineInfo>& infos, const Net* net,
    std::vector<std::shared_ptr<Tensor>>& allTensors) {
  std::vector<const Op*> ops;
  for (int i = 0; i < net->oplists()->size(); i++) {
    auto op = net->oplists()->GetAs<Op>(i);
    if (op->type() == OpType_Input) {
      continue;
    }
    ops.push_back(op);
  }
  initPipelineInfosFromOps(infos, ops, allTensors);
  setInputOutputForOps(allTensors, ops);
}
}  // namespace ace

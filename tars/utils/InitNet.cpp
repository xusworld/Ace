#include <unordered_map>

#include "InitNet.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/TensorUtils.hpp"
#include "glog/logging.h"
#include "half.hpp"

namespace tars {
bool needComputeOp(const Op* op) {
  if (op->type() != OpType_Input && op->type() != OpType_Const &&
      op->type() != OpType_TrainableParam) {
    return true;
  }
  return false;
}

// 考虑将 initConstTensors / initTensos 放在 Interpreter 中
bool initConstTensors(std::vector<std::shared_ptr<Tensor>>& tensors,
                      const Net* net, Device* defaultBackend, Status& code) {
  bool valid = true;
  tensors.resize(net->tensorName()->size());
  // Set up const
  LOG(INFO) << "net->tensorName()->size(): " << net->tensorName()->size();
  LOG(INFO) << "net->oplists()->size(): " << net->oplists()->size();
  for (int opIndex = 0; opIndex < net->oplists()->size(); ++opIndex) {
    // 获取 op
    auto op = net->oplists()->GetAs<Op>(opIndex);
    // 处理 constant op
    if (OpType_Const == op->type()) {
      CHECK(nullptr != op->outputIndexes());

      auto index = op->outputIndexes()->data()[0];
      LOG(INFO) << "index: " << index;
      tensors[index].reset(new Tensor);

      TensorUtils::getDescribe(tensors[index].get())->index = index;
      auto parameter = op->main_as_Blob();
      auto output = tensors[index].get();
      bool zeroShape = false;

      if (parameter->dims() != nullptr) {
        output->buffer().dimensions = parameter->dims()->size();
        for (int i = 0; i < output->buffer().dimensions; i++) {
          output->buffer().dim[i].extent = parameter->dims()->Get(i);
          if (output->length(i) <= 0) {
            zeroShape = true;
          }
        }
      } else {
        output->buffer().dimensions = 0;
      }

      if (parameter->dataType() == DataType_DT_HALF) {
        output->setType(DataType_DT_FLOAT);
      } else {
        output->setType(parameter->dataType());
      }

      TensorUtils::getDescribe(output)->dimensionFormat =
          parameter->dataFormat();
      TensorUtils::getDescribe(output)->usage =
          Tensor::InsideDescribe::CONSTANT;
      TensorUtils::getDescribe(output)->isMutable = false;
      TensorUtils::setLinearLayout(output);
      TensorUtils::getDescribe(output)->backend = defaultBackend;
      if (zeroShape) {
        continue;
      }
      auto res = defaultBackend->onAcquireBuffer(output, Device::STATIC);
      if (!res) {
        code = Status::ERROR();
        return false;
      }
      // 处理半精度数据
      if (parameter->dataType() == DataType_DT_HALF) {
        if (nullptr == parameter->uint8s()) {
          // Error half const
          code = Status::ERROR();
          return false;
        }
        auto outputPtr = output->host<float>();
        auto size = output->elementSize();
        half_float::half* src = nullptr;
        std::unique_ptr<half_float::half[]> tmp;
        if (USE_EXTERNAL_DATA(parameter)) {
          tmp.reset((new half_float::half[size]));
          src = tmp.get();
          OpCommonUtils::loadExternalDatas(defaultBackend,
                                           {reinterpret_cast<char*>(src)},
                                           parameter->external()->data());
        } else {
          src = (half_float::half*)parameter->uint8s()->data();
        }
        for (int i = 0; i < size; ++i) {
          outputPtr[i] = src[i];
        }
      } else {
        OpCommonUtils::loadBlobData(defaultBackend, op, output->host<char>(),
                                    output->size());
      }
    }
  }
  LOG(INFO) << "initConstTensors done.";
  return valid;
}

bool initTensors(std::vector<std::shared_ptr<Tensor>>& tensors,
                 const Net* net) {
  bool valid = true;
  // 删除 tensor descibe
  auto describes = net->extraTensorDescribe();

  LOG(INFO) << "initTensors| tensors.size(): " << tensors.size();
  std::vector<const TensorDescribe*> des(tensors.size());

  for (int i = 0; i < tensors.size(); ++i) {
    // Init all tensor except for const
    if (tensors[i].get() == nullptr) {
      tensors[i].reset(new Tensor);
      TensorUtils::getDescribe(tensors[i].get())->index = i;
    }
  }

  LOG(INFO) << "initTensors| describes: " << describes;
  if (describes) {
    LOG(INFO) << "initTensors| describes->size: " << describes->size();
    for (int i = 0; i < describes->size(); i++) {
      int index = describes->GetAs<TensorDescribe>(i)->index();
      // 将 tensor desc 写入 des 中
      des[index] = describes->GetAs<TensorDescribe>(i);
    }
  }

  for (int i = 0; i < tensors.size(); ++i) {
    if (des[i] != nullptr && des[i]->quantInfo()) {
      LOG(INFO) << "initTensors| handle qunat tensor if necessary";
      TensorUtils::getDescribe(tensors[i].get())
          ->quantAttr.reset(new QuantAttr);
      // 量化 tensor
      auto quant = TensorUtils::getDescribe(tensors[i].get())->quantAttr.get();
      quant->scale = des[i]->quantInfo()->scale();
      quant->zero = des[i]->quantInfo()->zero();
      quant->min = des[i]->quantInfo()->min();
      quant->max = des[i]->quantInfo()->max();
      // Don't copy datatype, it can be set by backend
    }
  }
  // Set Input Tensor, if the type of input is not the same with
  // ExtraTensorDescribe, use input parameter
  for (int opIndex = 0; opIndex < net->oplists()->size(); ++opIndex) {
    auto op = net->oplists()->GetAs<Op>(opIndex);
    if (OpType_Input == op->type()) {
      LOG(INFO) << "initTensors| handle model's input tensor";

      CHECK(nullptr != op->outputIndexes());
      CHECK(op->outputIndexes()->size() == 1);
      LOG(INFO) << "initTensors| op->name: " << op->name()->str();

      auto index = op->outputIndexes()->data()[0];
      LOG(INFO) << "initTensors| index: " << index;

      auto tensor = tensors[index].get();
      auto& tb = tensor->buffer();

      auto inputParam = op->main_as_Input();
      // 设置 input tensor 的各种属性
      if (auto idims = inputParam->dims()) {
        for (int i = 0; i < idims->size(); ++i) {
          int extent = idims->data()[i];
          LOG(INFO) << "initTensors| dim[" << i << "] = " << extent;
          // dim-0 is batch(when input batch is -1, set it to be 1, ignore
          // other dim)
          if (i == 0 && extent == -1) {
            extent = 1;
          }
          if (extent < 0) {
            valid = false;
          }
          tb.dim[i].extent = extent;
        }
        tb.dimensions = idims->size();
      } else {
        tb.dimensions = 0;
      }
      tensor->setType(inputParam->dtype());
      // TODO 通过 TensorUitls 设置 tensor 的属性，感觉十分没必要，修改这种设计
      TensorUtils::getDescribe(tensor)->dimensionFormat = inputParam->dformat();
      TensorUtils::setLinearLayout(tensor);
    }
  }
  if (net->usage() != Usage_INFERENCE_STATIC) {
    return valid;
  }
  // static model will set all tensors' shape
  for (int i = 0; i < describes->size(); i++) {
    int index = describes->GetAs<TensorDescribe>(i)->index();
    des[index] = describes->GetAs<TensorDescribe>(i);
  }
  for (int i = 0; i < tensors.size(); ++i) {
    if (TensorUtils::getDescribe(tensors[i].get())->usage !=
        Tensor::InsideDescribe::NORMAL) {
      // Const / Trainable Shape has been inited
      continue;
    }
    auto blob = des[i]->blob();
    auto& tb = tensors[i]->buffer();
    if (auto idims = blob->dims()) {
      for (int d = 0; d < idims->size(); d++) {
        tb.dim[d].extent = idims->Get(d);
      }
      tb.dimensions = idims->size();
    } else {
      tb.dimensions = 0;
    }
    tensors[i]->setType(blob->dataType());
  }

  for (int i = 0; i < tensors.size(); ++i) {
    auto blob = des[i]->blob();
    TensorUtils::getDescribe(tensors[i].get())->dimensionFormat =
        blob->dataFormat();
    if (auto regions = des[i]->regions()) {
      auto& regs = TensorUtils::getDescribe(tensors[i].get())->regions;
      TensorUtils::getDescribe(tensors[i].get())->memoryType =
          Tensor::InsideDescribe::MEMORY_BACKEND;
      regs.reserve(regions->size());
      for (int r = 0; r < regions->size(); r++) {
        auto region = regions->GetAs<Region>(r);
        Tensor::InsideDescribe::Region reg;
        reg.origin = tensors[region->origin()].get();
        reg.src.offset = region->src()->offset();
        reg.dst.offset = region->dst()->offset();
        for (int d = 0; d < 3; d++) {
          reg.size[d] = region->size()->data()[d];
          reg.src.stride[d] = region->src()->stride()->data()[d];
          reg.dst.stride[d] = region->dst()->stride()->data()[d];
        }
        regs.emplace_back(std::move(reg));
      }
    }
  }
  return valid;
}

void initPipelineInfosFromOps(
    std::vector<Schedule::OpCacheInfo>& infos, std::vector<const Op*>& ops,
    const std::vector<std::shared_ptr<Tensor>>& allTensors) {
  LOG(INFO) << "initPipelineInfosFromOps";
  LOG(INFO) << "initPipelineInfosFromOps| allTensors.size: "
            << allTensors.size();
  for (const Op* op : ops) {
    Schedule::OpCacheInfo opInfo;
    opInfo.op = op;
    if (op->outputIndexes() != nullptr) {
      auto data = op->outputIndexes()->data();
      // LOG(INFO) << "output size: " << op->outputIndexes()->size();
      for (int j = 0; j < op->outputIndexes()->size(); ++j) {
        opInfo.outputs.push_back(allTensors[data[j]].get());
      }
    }
    if (nullptr != op->inputIndexes()) {
      auto data = op->inputIndexes()->data();
      for (int j = 0; j < op->inputIndexes()->size(); ++j) {
        opInfo.inputs.push_back(allTensors[data[j]].get());
      }
    }
    if (needComputeOp(op)) {
      infos.emplace_back(std::move(opInfo));
    }
  }
}

void setInputOutputForOps(std::vector<std::shared_ptr<Tensor>>& allTensors,
                          const std::vector<const Op*>& ops, bool isStatic) {
  LOG(INFO) << "setInputOutputForOps";
  LOG(INFO) << "setInputOutputForOps| isStatic: " << isStatic;
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
    if (op->outputIndexes() != nullptr) {
      auto data = op->outputIndexes()->data();
      for (int j = 0; j < op->outputIndexes()->size(); ++j) {
        outputIndexes.insert(data[j]);
      }
    }

    if (op->inputIndexes() != nullptr) {
      auto data = op->inputIndexes()->data();
      for (int j = 0; j < op->inputIndexes()->size(); ++j) {
        inputIndexes.insert(data[j]);
      }
    }
    MNN_ASSERT(OpType_Input != op->type());
  }

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

  // 3. set usage for Tensor by index
  for (auto index : input) {
    auto des = TensorUtils::getDescribe(allTensors[index].get());
    if (des->usage == Tensor::InsideDescribe::CONSTANT) {
      continue;
    }

    des->usage = Tensor::InsideDescribe::INPUT;
  }

  for (auto index : output) {
    auto des = TensorUtils::getDescribe(allTensors[index].get());
    if (des->usage == Tensor::InsideDescribe::NORMAL) {
      des->usage = TensorUsage::OUTPUT;
    }
  }
}

void initPipelineInfosFromNet(
    std::vector<Schedule::OpCacheInfo>& infos, const Net* net,
    std::vector<std::shared_ptr<Tensor>>& allTensors) {
  LOG(INFO) << "initPipelineInfosFromNet";
  std::vector<const Op*> ops;

  for (int i = 0; i < net->oplists()->size(); i++) {
    auto op = net->oplists()->GetAs<Op>(i);
    LOG(INFO) << "op: " << op->name()->str();
    if (needComputeOp(op)) {
      ops.push_back(op);
    }
  }

  initPipelineInfosFromOps(infos, ops, allTensors);
  setInputOutputForOps(allTensors, ops);
}
}  // namespace tars

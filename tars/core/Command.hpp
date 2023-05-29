//
//  Command.hpp
//  MNN
//
//  Created by MNN on b'2020/07/28'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Command_hpp
#define Command_hpp
#include <memory>
#include <string>

#include "AutoStorage.h"
#include "core/tensor.h"
namespace tars {
struct Op;
class Operation;
class OperatorInfo;

struct Command : public RefCount {
  const Op* op;
  std::vector<Tensor*> workInputs;
  std::vector<Tensor*> inputs;
  std::vector<Tensor*> outputs;
  std::shared_ptr<BufferStorage> buffer;
  std::shared_ptr<Operation> execution;
  std::shared_ptr<OperatorInfo> info;
};

struct CommandBuffer {
  std::vector<SharedPtr<Command>> command;
  std::vector<std::shared_ptr<Tensor>> extras;
  bool hasWrap = false;
};
};  // namespace tars
#endif

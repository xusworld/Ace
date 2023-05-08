//
//  Command.hpp
//  MNN
//
//  Created by MNN on b'2020/07/28'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Command_hpp
#define Command_hpp
#include <ace/tensor.h>

#include <memory>
#include <string>
namespace ace {
struct Op;

struct Command {
  const Op* op;
  std::vector<Tensor*> inputs;
  std::vector<Tensor*> outputs;
  std::vector<uint8_t> buffer;  // storage for op
};
struct CommandBuffer {
  std::vector<Command> command;
  std::vector<std::shared_ptr<Tensor>> extras;
};
};  // namespace ace
#endif

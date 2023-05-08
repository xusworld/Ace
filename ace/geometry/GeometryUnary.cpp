//
//  GeometryUnary.cpp
//  MNN
//
//  Created by MNN on 2020/07/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"

namespace ace {

class GeometryUnary : public GeometryComputer {
 public:
  virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs,
                         const std::vector<Tensor*>& outputs, Context& context,
                         CommandBuffer& res) const override {
    // 断言输入输出数量
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    // 获取输入输出数值
    auto input = inputs[0];
    auto output = outputs[0];
    UnaryOpOperation unaryType;
    switch (op->type()) {
      case OpType_TanH:
        unaryType = UnaryOpOperation_TANH;
        break;
      case OpType_Sigmoid:
        unaryType = UnaryOpOperation_SIGMOID;
        break;
      default:
        break;
    }
    auto cmd = GeometryComputerUtils::makeUnary(unaryType, input, output);
    res.command.emplace_back(std::move(cmd));
    return true;
  }
};

static void _create() {
  std::shared_ptr<GeometryComputer> comp(new GeometryUnary);
  GeometryComputer::registerGeometryComputer(comp,
                                             {OpType_TanH, OpType_Sigmoid});
}

REGISTER_GEOMETRY(GeometryUnary, _create);

}  // namespace ace

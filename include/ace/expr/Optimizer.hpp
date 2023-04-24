//
//  Optimizer.hpp
//  MNN
//
//  Created by MNN on 2019/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef Optimizer_hpp
#define Optimizer_hpp
#include <ace/types.h>

#include <ace/expr/Expr.hpp>

namespace ace {
namespace Express {
class MNN_PUBLIC Optimizer {
 public:
  enum Device { CPU = 0, GPU = 1, OTHER = 2, AUTO = 3 };
  struct Config {
    Device device = CPU;
    DeviceType forwardType = DeviceType::ALL;
    int numThread = 4;
  };
  static std::shared_ptr<Optimizer> create(Config config);
  struct Cost {
    float compute;  // MFlops
    float memory;   // MB
  };
  class Parameters {
   public:
    Parameters(int n);
    virtual ~Parameters();

    float* get() const { return mValue; }
    int size() const { return mSize; }

   private:
    float* mValue;
    int mSize;
  };
  virtual std::shared_ptr<Parameters> onGetParameters(
      const std::vector<VARP>& outputs) {
    return nullptr;
  }

  // Given paramters and measure cost, the parameters must be the same as
  // onGetParameters
  virtual Cost onMeasure(const std::vector<VARP>& outputs,
                         std::shared_ptr<Parameters> parameters = nullptr) = 0;

  // Modify the output directly, the parameters must be the same as
  // onGetParameters
  virtual bool onExecute(const std::vector<VARP>& outputs,
                         std::shared_ptr<Parameters> parameters = nullptr) = 0;

  Optimizer() = default;
  virtual ~Optimizer() = default;
};
}  // namespace Express
}  // namespace ace
#endif

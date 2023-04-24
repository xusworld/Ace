//
//  MergeOptimizer.hpp
//  MNN
//
//  Created by MNN on 2019/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MergeOptimizer_hpp
#define MergeOptimizer_hpp

#include <ace/types.h>

#include <ace/expr/Optimizer.hpp>
namespace ace {
namespace Express {
class MergeOptimizer : public Optimizer {
 public:
  virtual ~MergeOptimizer() = default;
  MergeOptimizer(DeviceType type, int numberThread, BackendConfig* config);
  virtual Cost onMeasure(
      const std::vector<VARP>& outputs,
      std::shared_ptr<Parameters> parameters = nullptr) override;

  // Modify the output directly, the parameters must be the same as
  // onGetParameters
  virtual bool onExecute(
      const std::vector<VARP>& outputs,
      std::shared_ptr<Parameters> parameters = nullptr) override;

 private:
  BackendConfig mConfig;
  DeviceType mType;
  int mNumberThread;
};
};  // namespace Express
};  // namespace ace
#endif

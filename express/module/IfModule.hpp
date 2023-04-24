//
//  IfModule.hpp
//  MNN
//
//  Created by MNN on 2020/09/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef IfModule_hpp
#define IfModule_hpp

#include <ace/expr/Module.hpp>
namespace ace {
namespace Express {
class IfModule : public Module {
 public:
  virtual ~IfModule() {
    // Do nothing
  }
  virtual std::vector<Express::VARP> onForward(
      const std::vector<Express::VARP>& inputs) override;
  MNN_PUBLIC static IfModule* create(
      const Op* op, const std::map<std::string, SubGraph>& subGraph);

 private:
  IfModule() {}

  Module* clone(CloneContext* ctx) const override;

  // First mThen' index, Second: inputs's index
  std::vector<std::pair<int, int>> mInputForThen;

  // First mElse' index, Second: inputs's index
  std::vector<std::pair<int, int>> mInputForElse;

  std::vector<int> mOutputFromThen;
  std::vector<int> mOutputFromElse;

  std::shared_ptr<Module> mThen;
  std::shared_ptr<Module> mElse;
};
}  // namespace Express
}  // namespace ace

#endif /* IfModule_hpp */

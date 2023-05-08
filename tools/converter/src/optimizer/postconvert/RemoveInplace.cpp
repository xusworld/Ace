
#include "../PostTreatUtils.hpp"

class RemoveInplace : public PostConverter {
 public:
  virtual bool onExecute(std::unique_ptr<ace::NetT>& net) const override {
    // 遍历计算图上所有节点
    for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
      auto& op = *iter;

      // Inplace 算子必然是单输入单输出算子，判断该条件是否成立
      if (!PostTreatUtils::_isSingleInputOutput(op.get())) {
        continue;
      }
      // 判断输入 index 是否等于 输出 index
      if (op->inputIndexes[0] != op->outputIndexes[0]) {
        continue;
      }

      auto originIndex = op->inputIndexes[0];
      // 保存 op name
      net->tensorName.push_back(op->name);
      // 设置新的 index
      int newIndex = net->tensorName.size() - 1;
      op->outputIndexes[0] = newIndex;
      // 修改计算图上其他算子的指向
      for (auto subIter = iter + 1; subIter != net->oplists.end(); subIter++) {
        auto& subOp = *subIter;

        for (int i = 0; i < subOp->inputIndexes.size(); ++i) {
          if (subOp->inputIndexes[i] == originIndex) {
            subOp->inputIndexes[i] = newIndex;
          }
        }

        for (int i = 0; i < subOp->outputIndexes.size(); ++i) {
          if (subOp->outputIndexes[i] == originIndex) {
            subOp->outputIndexes[i] = newIndex;
          }
        }
      }
      net->tensorNumber = net->tensorName.size();
    }
    return true;
  }
};

static PostConverterRegister<RemoveInplace> __l("RemoveInplace");

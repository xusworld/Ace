//
//  SubGraphComplete.hpp
//  MNN
//
//  Created by MNN on 2020/06/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_CONVERTER_OPTIMIZER_SUBGRAPH_COMPLETE_HPP_
#define MNN_CONVERTER_OPTIMIZER_SUBGRAPH_COMPLETE_HPP_

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ace/expr/Expr.hpp"
#include "ace_generated.h"

namespace ace {
namespace Express {

struct OptimizeContext {
  std::vector<ace::SubGraphProtoT*> subgraphs;
  bool is_training;
  bool verbose;
  FrontendFramework source;

  std::vector<SubGraphProtoT*> completed_subgraphs;

  using NetTPtr = std::unique_ptr<ace::NetT>;
  template <typename K, typename V>
  using HashMap = std::unordered_map<K, V>;

  // NetTPtr (*RunOptimize)(NetTPtr&, const HashMap<std::string, VARP>&);
  std::function<NetTPtr(NetTPtr&,  // NOLINT
                        const HashMap<std::string, VARP>&)>
      RunOptimize;
};

SubGraphProtoT* FindSubGraphByName(
    const std::vector<SubGraphProtoT*>& subgraphs,
    const std::string& subgraph_name);

bool CompleteSubGraph(const std::unordered_map<std::string, VARP>& inputs,
                      const SubGraphProtoT* subgraph);

}  // namespace Express
}  // namespace ace

#endif  // MNN_CONVERTER_OPTIMIZER_SUBGRAPH_COMPLETE_HPP_

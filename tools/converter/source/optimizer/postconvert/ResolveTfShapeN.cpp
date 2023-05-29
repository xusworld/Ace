//
//  ResolveTfShapeN.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
#include "flatbuffers/util.h"

class ResolveTfShapeN : public PostConverter {
 public:
  virtual bool onExecute(std::unique_ptr<tars::NetT>& net) const override {
    if (net->sourceType != tars::NetSource_TENSORFLOW) {
      return true;
    }

    std::set<tars::OpT*> readyToDelete;

    for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
      const auto op = iter->get();
      if (op->type != tars::OpType_Extra) {
        iter++;
        continue;
      }
      auto attr = op->main.AsExtra();
      auto& optype = attr->type;
      if (optype != "ShapeN") {
        iter++;
        continue;
      }
      int shapeNumber = 1;
      const int attrSize = attr->attr.size();
      for (int k = 0; k < attrSize; ++k) {
        auto& key = attr->attr[k]->key;
        if (key == "N") {
          shapeNumber = attr->attr[k]->i;
        }
      }
      readyToDelete.insert(op);
      // expand ShapeN to N Shapes
      // insert N Shape before ShapeN, then delete the shapeN
      for (int i = 0; i < shapeNumber; ++i) {
        std::unique_ptr<tars::OpT> curShape(new tars::OpT);
        curShape->name = op->name + flatbuffers::NumToString(i);
        curShape->type = tars::OpType_Shape;
        curShape->main.value = nullptr;
        curShape->inputIndexes.push_back(op->inputIndexes[i]);
        curShape->outputIndexes.push_back(op->outputIndexes[i]);
        iter = net->oplists.insert(iter, std::move(curShape));
        iter++;
      }

      iter++;
    }

    for (auto op : readyToDelete) {
      PostTreatUtils::_removeOpInNet(op, net.get());
    }

    return true;
  }
};

static PostConverterRegister<ResolveTfShapeN> __shapen("ResolveTfShapeN");

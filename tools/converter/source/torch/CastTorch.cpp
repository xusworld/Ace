//
//  CastTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/07/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(CastTorch);

tars::OpType CastTorch::opType() { return tars::OpType_Cast; }
tars::OpParameter CastTorch::type() { return tars::OpParameter_CastParam; }
std::vector<int> CastTorch::inputTensorIdx() { return {0}; }

void CastTorch::run(tars::OpT* dstOp, const torch::jit::Node* node,
                    TorchScope* scope) {
  auto param = new tars::CastParamT;
  std::string opType = getRealOpType(node);
  if (opType == "to" || opType == "as_tensor") {
    /*
     scalar_type_to_pytorch_type = [
         torch.uint8,        # 0
         torch.int8,         # 1
         torch.short,        # 2
         torch.int,          # 3
         torch.int64,        # 4
         torch.half,         # 5
         torch.float,        # 6
         torch.double,       # 7
         torch.complex32,    # 8
         torch.complex64,    # 9
         torch.complex128,   # 10
         torch.bool,         # 11
     ]
     */
    static std::vector<tars::DataType> gMaps{
        tars::DataType_DT_UINT8,   tars::DataType_DT_INT8,
        tars::DataType_DT_INT16,   tars::DataType_DT_INT32,
        tars::DataType_DT_INT64,   tars::DataType_DT_HALF,
        tars::DataType_DT_FLOAT,   tars::DataType_DT_DOUBLE,
        tars::DataType_DT_INVALID, tars::DataType_DT_INVALID,
        tars::DataType_DT_INVALID, tars::DataType_DT_BOOL,
    };
    param->dstT = gMaps[getValue<int64_t>(node->input(1))];
  } else if (opType == "type_as") {
    auto type = node->input(1)->type()->cast<at::TensorType>();
    if (type) {
      auto scalarType = type->scalarType().value_or(at::ScalarType::Float);
      param->dstT = ScalarType2Dtype(scalarType);
    } else {
      param->dstT = tars::DataType_DT_FLOAT;
    }
  } else {
    static std::map<std::string, tars::DataType> gMaps{
        {"Int", tars::DataType_DT_INT32},
        {"IntImplicit", tars::DataType_DT_INT32},
        {"Bool", tars::DataType_DT_BOOL},
        {"Float", tars::DataType_DT_FLOAT},
        {"FloatImplicit", tars::DataType_DT_FLOAT},
    };
    param->dstT = gMaps[opType];
  }
  dstOp->main.value = param;
}

REGISTER_CONVERTER(CastTorch, Int);
REGISTER_CONVERTER(CastTorch, IntImplicit);
REGISTER_CONVERTER(CastTorch, Bool);
REGISTER_CONVERTER(CastTorch, Float);
REGISTER_CONVERTER(CastTorch, FloatImplicit);
REGISTER_CONVERTER(CastTorch, to);
REGISTER_CONVERTER(CastTorch, type_as);
REGISTER_CONVERTER(CastTorch, as_tensor);

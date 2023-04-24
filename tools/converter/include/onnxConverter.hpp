//
//  onnxConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ONNXCONVERTER_HPP
#define ONNXCONVERTER_HPP
#include <ace/MNNDefine.h>

#include "ace_generated.h"

namespace ace {
namespace parser {

int OnnxToAceModel(const std::string inputModel, const std::string bizCode,
                   std::unique_ptr<ace::NetT>& netT);
}  // namespace parser
}  // namespace ace

#endif  // ONNXCONVERTER_HPP

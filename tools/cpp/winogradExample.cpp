//
//  winogradExample.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include <stdlib.h>
#include <string.h>

#include "math/Matrix.hpp"
#include "math/WingoradGenerater.hpp"

int main(int argc, const char* argv[]) {
  int unit = ::atoi(argv[1]);
  int kernelSize = ::atoi(argv[2]);
  float interp = 0.5f;
  if (argc > 3) {
    interp = ::atof(argv[3]);
  }
  tars::Math::WinogradGenerater generater(unit, kernelSize, interp);
  auto a = generater.A();
  auto b = generater.B();
  auto g = generater.G();
  MNN_PRINT("A=\n");
  tars::Math::Matrix::print(a.get());
  MNN_PRINT("B=\n");
  tars::Math::Matrix::print(b.get());
  MNN_PRINT("G=\n");
  tars::Math::Matrix::print(g.get());
  return 0;
}

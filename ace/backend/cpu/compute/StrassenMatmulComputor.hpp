//
//  StrassenMatmulComputor.hpp
//  MNN
//
//  Created by MNN on 2019/02/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef StrassenMatmulComputor_hpp
#define StrassenMatmulComputor_hpp

#include <functional>

#include "core/Backend.hpp"
namespace ace {
/**
 Based on
 Boyer, B., Dumas, J.-G., Pernet, C., & Zhou, W. (2007). Memory efficient
 scheduling of Strassen-Winogradʼs matrix multiplication algorithm. Proceedings
 of the 2009 international symposium on Symbolic and algebraic computation ISSAC
 09, 55. ACM Press. Retrieved from http://arxiv.org/abs/0707.2347

 Use Table 2
 */
class StrassenMatrixComputor {
 public:
  StrassenMatrixComputor(Backend* bn, bool multithread, int maxDepth);
  virtual ~StrassenMatrixComputor();

  /*
   It's assume that:
   A is a matrix where each element is a (4,1) vector : lC4, e, 4
   B is a matrix where each element is a (hP,1) vector : h, l, hP
   inputs[0] is the transpose of A: AT, inputs[1] is the transpose of B: BT
   outputs[0] is the transpose of C: CT
   C is a matrix where each element is a (4,1) vector, the same as A : hC4, e, 4

   if (inputs.size() > 2) {
      inputs[2] is origin CO: CT
      CO can be the same same as C or broadcast in lenght(1): hC4, e, 4 or hC4,
   1, 4
   }
   Compute: C = alpha * AB + beta * CO , alpha must be 1.0f

   postParameters:
   0: alpha
   1: beta
   2: min
   3: max

   if (postParameters.empty()) {
      alpha = 1.0f
      beta = 0.0f;
      min = -FLT_MAX
      max = FLT_MAX
   }
   */
  ErrorCode onEncode(const std::vector<Tensor*>& inputs,
                     const std::vector<Tensor*>& outputs,
                     const std::vector<float>& postParameters = {}, int l = 0,
                     int h = 0);

  void onExecute(const uint8_t* AT = nullptr, const uint8_t* BT = nullptr,
                 const uint8_t* COT = nullptr, uint8_t* CT = nullptr);

  void onReset();

 protected:
  Backend* backend() const { return mBackend; }

 private:
  struct MatrixInfo {
    int stackIndex;
    int offsetBytes;
    int lineStrideBytes;
  };
  ErrorCode _generateMatMul(int e, int l, int h, const MatrixInfo& AT,
                            const MatrixInfo& BT, const MatrixInfo& CT,
                            const MatrixInfo& COT, int currentDepth,
                            const std::vector<float>& postParameters);
  ErrorCode _generateTrivalMatMul(int e, int l, int h, const MatrixInfo& AT,
                                  const MatrixInfo& BT, const MatrixInfo& CT,
                                  const MatrixInfo& COT,
                                  const std::vector<float>& postParameters);
  ErrorCode _generateBasicMatMul(int e, int l, int h, const MatrixInfo& AT,
                                 const MatrixInfo& BT, const MatrixInfo& CT,
                                 const MatrixInfo& COT,
                                 const std::vector<float>& postParameters);

  std::vector<std::pair<std::function<void(int tId)>, int>> mFunctions;
  int mMaxDepth;
  bool mSupportMultiThread;

  Backend* mBackend;

  std::vector<uint8_t*> mStack;
};
}  // namespace ace

#endif /* StrassenMatmulComputor_hpp */
//
//  MergeScaleToConvolution.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
#include "MergeToConvolution.hpp"

using namespace tars;

class MergeScaleToConvolution : public MergeToConvolution {
 public:
  bool merge2Convolution(const tars::OpT* inplaceOp,
                         tars::OpT* convolutionOp) const {
    const auto& convCommon = convolutionOp->main.AsConvolution2D()->common;
    if (convCommon->relu || convCommon->relu6 ||
        convolutionOp->inputIndexes.size() > 1) {
      return false;
    }

    if (inplaceOp->type == tars::OpType_Scale) {
      std::vector<float> alpha;
      std::vector<float> bias;

      bias = inplaceOp->main.AsScale()->biasData;
      alpha = inplaceOp->main.AsScale()->scaleData;

      auto conv2D = convolutionOp->main.AsConvolution2D();
      int outputCount = conv2D->common->outputCount;
      for (int i = 0; i < outputCount; ++i) {
        conv2D->bias[i] = conv2D->bias[i] * alpha[i] + bias[i];
      }

      if (nullptr != conv2D->quanParameter.get()) {
        for (int i = 0; i < outputCount; ++i) {
          conv2D->quanParameter->alpha[i] *= alpha[i];
        }
      } else {
        int weightPartSize = conv2D->weight.size() / outputCount;
        if (convolutionOp->type == OpType_Deconvolution) {
          int inputCount = conv2D->weight.size() / outputCount /
                           conv2D->common->kernelX / conv2D->common->kernelY;
          for (int i = 0; i < inputCount; ++i) {
            auto dstPos = i * outputCount * conv2D->common->kernelY *
                          conv2D->common->kernelX;
            for (int j = 0; j < outputCount; ++j) {
              auto dstPosJ = dstPos + j * conv2D->common->kernelY *
                                          conv2D->common->kernelX;
              float a = alpha[j];
              for (int k = 0;
                   k < conv2D->common->kernelY * conv2D->common->kernelX; ++k) {
                conv2D->weight[dstPosJ + k] *= a;
              }
            }
          }
        } else {
          for (int i = 0; i < outputCount; ++i) {
            float a = alpha[i];
            for (int j = 0; j < weightPartSize; ++j) {
              conv2D->weight[i * weightPartSize + j] *= a;
            }
          }
        }
      }
      return true;
    }
    return false;
  }

  bool merge2Convolution3D(const tars::OpT* inplaceOp,
                           tars::OpT* convolutionOp) const {
    const auto& convCommon = convolutionOp->main.AsConvolution3D()->common;
    if (convCommon->relu || convCommon->relu6) {
      return false;
    }

    if (inplaceOp->type == tars::OpType_Scale) {
      std::vector<float> alpha;
      std::vector<float> bias;

      bias = inplaceOp->main.AsScale()->biasData;
      alpha = inplaceOp->main.AsScale()->scaleData;

      auto conv3D = convolutionOp->main.AsConvolution3D();
      int outputCount = conv3D->common->outputCount;
      for (int i = 0; i < outputCount; ++i) {
        conv3D->bias[i] = conv3D->bias[i] * alpha[i] + bias[i];
      }

      int weightPartSize = conv3D->weight.size() / outputCount;
      for (int i = 0; i < outputCount; ++i) {
        float a = alpha[i];
        for (int j = 0; j < weightPartSize; ++j) {
          conv3D->weight[i * weightPartSize + j] *= a;
        }
      }
      return true;
    }
    return false;
  }
};
static PostConverterRegister<MergeScaleToConvolution> __l(
    "MergeScaleToConvolution");

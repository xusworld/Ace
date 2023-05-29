//
//  revertMNNModel.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include <stdlib.h>
#include <string.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>

#include "common/CommonCompute.hpp"
#include "common/MemoryFormater.h"
#include "revertMNNModel.hpp"

Revert::Revert(const char* originalModelFileName) {
  std::ifstream inputFile(originalModelFileName, std::ios::binary);
  inputFile.seekg(0, std::ios::end);
  const auto size = inputFile.tellg();
  inputFile.seekg(0, std::ios::beg);

  char* buffer = new char[size];
  inputFile.read(buffer, size);
  inputFile.close();
  mMNNNet = tars::UnPackNet(buffer);
  delete[] buffer;
  MNN_ASSERT(mMNNNet->oplists.size() > 0);
}

Revert::~Revert() {}

void* Revert::getBuffer() const {
  return reinterpret_cast<void*>(mBuffer.get());
}

const size_t Revert::getBufferSize() const { return mBufferSize; }

void Revert::packMNNNet() {
  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = tars::Net::Pack(builder, mMNNNet.get());
  builder.Finish(offset);
  mBufferSize = builder.GetSize();
  mBuffer.reset(new uint8_t[mBufferSize], std::default_delete<uint8_t[]>());
  ::memcpy(mBuffer.get(), builder.GetBufferPointer(), mBufferSize);
  mMNNNet.reset();
}

void Revert::initialize(float spasity, int sparseBlockOC, bool rewrite) {
  if (mMNNNet->bizCode == "benchmark" || rewrite) {
    randStart();
    for (auto& op : mMNNNet->oplists) {
      const auto opType = op->type;
      switch (opType) {
        case tars::OpType_Convolution:
        case tars::OpType_Deconvolution:
        case tars::OpType_ConvolutionDepthwise: {
          auto param = op->main.AsConvolution2D();
          auto& convCommon = param->common;
          const int weightReduceStride = convCommon->kernelX *
                                         convCommon->kernelY *
                                         convCommon->inputCount;
          const int oc = convCommon->outputCount / convCommon->group;
          param->weight.resize(oc * weightReduceStride);
          ::memset(param->weight.data(), 0,
                   param->weight.size() * sizeof(float));
          size_t weightNNZElement, weightBlockNumber = 0;
          tars::CommonCompute::fillRandValueAsSparsity(
              weightNNZElement, weightBlockNumber, param->weight.data(), oc,
              weightReduceStride, spasity, sparseBlockOC);

          tars::AttributeT* arg1(new tars::AttributeT);
          arg1->key = "sparseBlockOC";
          arg1->i = sparseBlockOC;

          tars::AttributeT* arg2(new tars::AttributeT);
          arg2->key = "sparseBlockKernel";
          arg2->i = 1;

          tars::AttributeT* arg3(new tars::AttributeT);
          arg3->key = "NNZElement";
          arg3->i = weightNNZElement;

          tars::AttributeT* arg4(new tars::AttributeT);
          arg4->key = "blockNumber";
          arg4->i = weightBlockNumber;

          flatbuffers::FlatBufferBuilder builder;
          std::vector<flatbuffers::Offset<tars::Attribute>> argsVector;
          auto sparseArg1 = tars::CreateAttribute(builder, arg1);
          auto sparseArg2 = tars::CreateAttribute(builder, arg2);
          auto sparseArg3 = tars::CreateAttribute(builder, arg3);
          auto sparseArg4 = tars::CreateAttribute(builder, arg4);

          argsVector.emplace_back(sparseArg1);
          argsVector.emplace_back(sparseArg2);
          argsVector.emplace_back(sparseArg3);
          argsVector.emplace_back(sparseArg4);

          auto sparseArgs =
              builder.CreateVectorOfSortedTables<tars::Attribute>(&argsVector);
          tars::SparseAlgo prune_algo_type;
          if (sparseBlockOC == 4) {
            prune_algo_type = tars::SparseAlgo_SIMD_OC;
          } else {
            prune_algo_type = tars::SparseAlgo_RANDOM;
          }
          auto sparseCom =
              tars::CreateSparseCommon(builder, prune_algo_type, sparseArgs);
          builder.Finish(sparseCom);
          auto sparseComPtr = flatbuffers::GetRoot<tars::SparseCommon>(
                                  builder.GetBufferPointer())
                                  ->UnPack();
          param->sparseParameter.reset(sparseComPtr);

          param->bias.resize(convCommon->outputCount);
          ::memset(param->bias.data(), 0, param->bias.size() * sizeof(float));
          break;
        }
        case tars::OpType_Scale: {
          auto param = op->main.AsScale();
          param->biasData.resize(param->channels);
          param->scaleData.resize(param->channels);
          fillRandValue(param->scaleData.data(), param->channels);
          fillRandValue(param->biasData.data(), param->channels);
          break;
        }
        default:
          break;
      }
    }
  }

  packMNNNet();
}

void Revert::fillRandValue(float* data, size_t size) {
  unsigned int seed = 1000;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform_dist(-2, 2);

  for (size_t i = 0; i < size; i++) {
    *data = uniform_dist(rng);
  }
  return;
}

void Revert::randStart() {}

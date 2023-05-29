//
//  AddSparseInfo.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonUtils.hpp"
#include "backend/cpu/compute/SparseConvolutionTiledExecutor.hpp"
#include "common/CommonCompute.hpp"

using namespace tars;

void AddSparseInfo(std::unique_ptr<tars::OpT>& op,
                   Compression::Pipeline proto) {
  auto prune_algo_type = tars::SparseAlgo_RANDOM;
  int sparseBlockOC = 1;
  int sparseBlockKernel = 1;

  for (const auto& algo : proto.algo()) {
    if (algo.type() == Compression::CompressionAlgo::PRUNE) {
      auto prune_type = algo.prune_params().type();
      prune_algo_type = tars::SparseAlgo(prune_type);
      if (prune_type == Compression::PruneParams_PruneType_SIMD_OC) {
        sparseBlockOC =
            algo.prune_params().simd_oc_pruner_params().oc_blocks(0);
      }
    }
  }

  const auto opType = op->type;
  switch (opType) {
    case tars::OpType_Convolution:
    case tars::OpType_ConvolutionDepthwise: {
      auto param = op->main.AsConvolution2D();
      if (param->weight.empty()) {
        return;
      }

      size_t weightSize = param->weight.size();
      size_t biasSize = param->bias.size();
      size_t weightNNZElement, weightBlockNumber = 0;
      CommonCompute::statisticWeightSparsity(
          weightNNZElement, weightBlockNumber, param->weight.data(), biasSize,
          weightSize / biasSize, sparseBlockOC);
      float sparsity = 1. - double(weightNNZElement) / weightSize;
      // MNN_PRINT(" opname [%s] sparsity is:%f\n", op->name.c_str(), sparsity);
      if (!SparseConvolutionTiledExecutor::shouldUseSparseConvolution(
              sparsity, sparseBlockOC)) {
        return;
      }

      tars::AttributeT* arg1(new tars::AttributeT);
      arg1->key = "sparseBlockOC";
      arg1->i = sparseBlockOC;

      tars::AttributeT* arg2(new tars::AttributeT);
      arg2->key = "sparseBlockKernel";
      arg2->i = sparseBlockKernel;

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
      auto sparseCom =
          tars::CreateSparseCommon(builder, prune_algo_type, sparseArgs);
      builder.Finish(sparseCom);
      auto sparseComPtr =
          flatbuffers::GetRoot<tars::SparseCommon>(builder.GetBufferPointer())
              ->UnPack();

      param->sparseParameter.reset(sparseComPtr);

      break;
    }
    default:
      break;
  }
};

void addSparseInfo(std::unique_ptr<tars::NetT>& netT,
                   tars::Compression::Pipeline proto) {
  for (auto& op : netT->oplists) {
    AddSparseInfo(op, proto);
  }
  for (auto& subgraph : netT->subgraphs) {
    for (auto& op : subgraph->nodes) {
      AddSparseInfo(op, proto);
    }
  }
}

#include <stdio.h>

#include "../onnx_node_parser_manager.h"

namespace ace {
namespace parser {

DECLARE_ONNX_NODE_PARSER(GemmOnnx);

ace::OpType GemmOnnx::opType() { return ace::OpType_InnerProduct; }
ace::OpParameter GemmOnnx::type() { return ace::OpParameter_InnerProduct; }

void GemmOnnx::parse(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                     std::vector<const onnx::TensorProto*> initializers) {
  const int size = initializers.size();
  DCHECK(size <= 2 && size >= 1) << "Gemm Input ERROR!";
  auto gemmParam = new ace::InnerProductT;

  bool transA = false;
  bool transB = false;
  float alpha = 1.0f;
  float beta = 1.0f;

  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "transA") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT)
          << "Node Attribute ERROR";
      transA = static_cast<bool>(attributeProto.i());
    } else if (attributeName == "transB") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT)
          << "Node Attribute ERROR";
      transB = static_cast<bool>(attributeProto.i());
    } else if (attributeName == "alpha") {
      DCHECK(attributeProto.type() ==
             ::onnx::AttributeProto_AttributeType_FLOAT)
          << "Node Attribute ERROR";
      alpha = attributeProto.f();
    } else if (attributeName == "beta") {
      DCHECK(attributeProto.type() ==
             ::onnx::AttributeProto_AttributeType_FLOAT)
          << "Node Attribute ERROR";
      beta = attributeProto.f();
    }
  }

  // TODO, implement (alpha * A * B + beta * C), now (A * B + C)
  DCHECK(1 == alpha);
  DCHECK(1 == beta);

  DCHECK(!transA) << "Now GEMM not support transpose the intput tensor!";

  int weightSize = 1;
  const auto weightProto = initializers[0];
  DCHECK(2 == weightProto->dims_size()) << "Gemm weight dimensions should be 2";
  int bK = weightProto->dims(0);
  int bN = weightProto->dims(1);

  if (transB) {
    int temp = bK;
    bK = bN;
    bN = temp;
  }
  for (int i = 0; i < weightProto->dims_size(); ++i) {
    weightSize *= weightProto->dims(i);
  }

  std::vector<float> weightContainer(weightSize);
  auto weightPtr = weightContainer.data();

  if (weightProto->float_data_size() != 0) {
    for (int i = 0; i < weightSize; ++i) {
      weightPtr[i] = weightProto->float_data(i);
    }
  } else if (weightProto->raw_data().data()) {
    ::memcpy(weightPtr,
             reinterpret_cast<const float*>(weightProto->raw_data().data()),
             weightSize * sizeof(float));
  } else {
    DLOG(ERROR) << "ERROR";
  }

  auto weightBlob = new ace::BlobT;
  // tranpose weight if necessary
  weightBlob->dims.resize(2);
  weightBlob->dims[0] = bK;
  weightBlob->dims[1] = bN;
  if (transB) {
    auto& weightVector = weightBlob->float32s;
    weightVector.resize(weightSize);
    for (int i = 0; i < bK; ++i) {
      for (int j = 0; j < bN; ++j) {
        weightVector[i * bN + j] = weightContainer[j * bK + i];
      }
    }
  } else {
    weightBlob->float32s = weightContainer;
  }
  gemmParam->weight = weightContainer;

  // bias
  std::vector<float> biasContainer(bN);
  const auto biasProto = size == 2 ? initializers[1] : nullptr;
  if (biasProto) {
    int biasSize = 1;
    DCHECK(1 == biasProto->dims_size()) << "Gemm bias dimension should be 1";
    for (int i = 0; i < biasProto->dims_size(); ++i) {
      biasSize *= biasProto->dims(i);
    }
    // TODO,support broadcast add for( + C)
    DCHECK(bN == biasSize) << "Gemm Now not support for broadcast mode(+ C)";
    auto biasPtr = biasContainer.data();
    if (biasProto->float_data_size() != 0) {
      for (int i = 0; i < biasSize; ++i) {
        biasPtr[i] = biasProto->float_data(i);
      }
    } else if (biasProto->raw_data().data()) {
      ::memcpy(biasPtr,
               reinterpret_cast<const float*>(biasProto->raw_data().data()),
               biasSize * sizeof(float));
    } else {
      DLOG(ERROR) << "ERROR";
    }
  }
  gemmParam->bias = biasContainer;

  gemmParam->outputCount = bN;
  gemmParam->axis = 1;
  gemmParam->transpose = false;
  gemmParam->biasTerm = 1;

  dstOp->main.value = gemmParam;
}

// REGISTER_ONNX_NODE_PARSER(GemmOnnx, Gemm);

DECLARE_ONNX_NODE_PARSER(MatMulOnnx);

ace::OpType MatMulOnnx::opType() { return ace::OpType_MatMul; }

ace::OpParameter MatMulOnnx::type() { return ace::OpParameter_MatMul; }

void MatMulOnnx::parse(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                       std::vector<const onnx::TensorProto*> initializers) {
  CHECK(2 == onnxNode->input_size()) << "ONNX Matmul input error!";
  auto param = new ace::MatMulT;
  param->T = ace::DataType_DT_FLOAT;

  dstOp->main.value = param;
}

REGISTER_ONNX_NODE_PARSER(MatMulOnnx, MatMul);

}  // namespace parser
}  // namespace ace
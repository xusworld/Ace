//
//  ShapeRNNSequenceGRU.cpp
//  MNN
//
//  Created by MNN on 2019/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/TensorUtils.hpp"
#include "shape/SizeComputer.hpp"
namespace ace {

class RNNSequenceGRUComputer : public SizeComputer {
 public:
  virtual bool onComputeSize(
      const ace::Op* op, const std::vector<Tensor*>& inputs,
      const std::vector<Tensor*>& outputs) const override {
    MNN_ASSERT(6 <= inputs.size());
    MNN_ASSERT(1 <= outputs.size());

    auto input = inputs[0];  // typically onnx input shape: {sequenceLength,
                             // batchSize, inputLength}
    auto output = outputs[0];

    const auto rnnParam = op->main_as_RNNParam();
    const int numUnits = rnnParam->numUnits();
    bool keepAllOutputs = rnnParam->keepAllOutputs();
    bool isBidirectionalRNN = rnnParam->isBidirectionalRNN();

    // input->printShape();

    // MNN_ASSERT(2 == rnnParam->fwGateWeight()->dims()->size());
    // MNN_ASSERT(2 * numUnits == rnnParam->fwGateWeight()->dims()->data()[1]);
    // MNN_ASSERT((input->length(2) + numUnits) ==
    // rnnParam->fwGateWeight()->dims()->data()[0]);
    output->buffer().type = halide_type_of<float>();
    TensorUtils::getDescribe(output)->dimensionFormat =
        TensorUtils::getDescribe(input)->dimensionFormat;

    if (keepAllOutputs) {
      TensorUtils::setShape(output, {input->length(0), isBidirectionalRNN + 1,
                                     input->length(1), numUnits});
      // output shape: {sequenceLength, numDirection, batchSize, inputLength}
      // !!caution: onnx model graph some time would squeeze the ‘1 dim’ in
      // output 'numDirection', we should keep numDirection index at 1, but, the
      // typical memory layout of input tensor in CPURNNSequenceGRU.cpp is
      // {batch, sequenceLength, inputLength}, there is mismatch here when batch
      // or sequence is not 1
      output->buffer().type = input->buffer().type;

    } else {  // only keep the last hidden layer sequence
      TensorUtils::setShape(
          output, {1, isBidirectionalRNN + 1, input->length(1), numUnits});
      output->buffer().type = input->buffer().type;
    }

    // output->printShape();

    return true;
  }
};

REGISTER_SHAPE(RNNSequenceGRUComputer, OpType_RNNSequenceGRU);
}  // namespace ace
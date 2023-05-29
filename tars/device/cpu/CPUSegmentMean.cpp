#include <cmath>

#include "MNN_generated.h"
#include "core/Concurrency.h"
#include "core/operation.h"
#include "device/cpu/CPUDevice.h"

namespace tars {

class CPUSegmentMean : public Operation {
  int mDim;

 public:
  explicit CPUSegmentMean(const tars::Op *op, Device *backend);
  Status onResize(const std::vector<Tensor *> &inputs,
                  const std::vector<Tensor *> &outputs) override;
  Status onExecute(const std::vector<Tensor *> &inputs,
                   const std::vector<Tensor *> &outputs) override;
};  // class CPUSegmentMean;

CPUSegmentMean::CPUSegmentMean(const tars::Op *op, Device *backend)
    : Operation(backend) {
  mDim = 1;
  return;
}  // CPUSegmentMean

Status CPUSegmentMean::onExecute(const std::vector<Tensor *> &inputs,
                                 const std::vector<Tensor *> &outputs) {
  auto data = inputs[0];
  auto segmentIds = inputs[1];
  int seq_len = data->length(0);
  int k = 0;
  int c = 0;
  int dim = mDim;
  memset((void *)outputs[0]->host<float>(), 0, outputs[0]->size());
  for (int i = 0; i < seq_len; i++) {
    if (segmentIds->host<int>()[i] - k == 1) {
      for (int j = 0; j < dim; j++) {
        outputs[0]->host<float>()[k * dim + j] /= c;
      }
      k += 1;
      c = 0;
    }
    for (int j = 0; j < dim; j++) {
      outputs[0]->host<float>()[k * dim + j] +=
          data->host<float>()[i * dim + j];
    };
    c += 1;
    if (i == seq_len - 1) {
      for (int j = 0; j < dim; j++) {
        outputs[0]->host<float>()[k * dim + j] /= c;
      }
    }
  }
  return Status::OK();
}

Status CPUSegmentMean::onResize(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs) {
  auto data = inputs[0];
  mDim = 1;
  for (int i = 1; i < data->buffer().dimensions; i++) {
    mDim *= data->length(i);
  }
  return Status::OK();
}

class CPUSegmentMeanCreator : public CPUDevice::Creator {
 public:
  Operation *onCreate(const std::vector<Tensor *> &inputs,
                      const std::vector<Tensor *> &outputs, const tars::Op *op,
                      Device *backend) const override {
    return new CPUSegmentMean(op, backend);
  }
};

REGISTER_CPU_OP_CREATOR(CPUSegmentMeanCreator, OpType_Segment);

}  // namespace tars

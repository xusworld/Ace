//
//  CPUInterp.cpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>

#include "device/cpu/CPUDevice.h"
#include "device/cpu/CPUInterp.hpp"
#include "device/cpu/CPUResize.hpp"

namespace tars {

static int CLAMP(int v, int min, int max) {
  if ((v) < min) {
    (v) = min;
  } else if ((v) > max) {
    (v) = max;
  }
  return v;
}

CPUInterp::CPUInterp(Device *backend, int resizeType, float widthScale,
                     float heightScale, float widthOffset, float heightOffset)
    : CPUResizeCommon(backend),
      mResizeType(resizeType),
      mWidthScale(widthScale),
      mHeightScale(heightScale),
      mWidthOffset(widthOffset),
      mHeightOffset(heightOffset) {
  // nothing to do
}

CPUInterp::~CPUInterp() {
  if (mInit && mResizeType == 2) {
    backend()->onReleaseBuffer(&mWidthPosition, Device::STATIC);
    backend()->onReleaseBuffer(&mWidthFactor, Device::STATIC);
    backend()->onReleaseBuffer(&mHeightPosition, Device::STATIC);
    backend()->onReleaseBuffer(&mHeightFactor, Device::STATIC);
  }
}

Status CPUInterp::onExecute(const std::vector<Tensor *> &inputs,
                            const std::vector<Tensor *> &outputs) {
  auto &input = inputs[0]->buffer();
  auto &output = outputs[0]->buffer();

  if (mResizeType == 1) {
    // Nearstneighbor
    CPUResizeNearestneighborC4(input, output, mWidthScale, mHeightScale,
                               mWidthOffset, mHeightOffset);
  } else if (mResizeType == 2) {
    // bilinear
    CPUResizeBilinearC4(input, output, mWidthPosition.host<int>(),
                        mWidthFactor.host<float>(), mHeightPosition.host<int>(),
                        mHeightFactor.host<float>(), mLineBuffer.host<float>(),
                        ((CPUDevice *)backend())->threadNumber());
  } else if (mResizeType == 3) {
    // cubic
    CPUResizeCubicC4(input, output, mWidthScale, mHeightScale, mWidthOffset,
                     mHeightOffset);
  } else if (mResizeType == 4) {
    // Nearstneighbor
    CPUResizeNearestneighborRoundC4(input, output, mWidthScale, mHeightScale,
                                    mWidthOffset, mHeightOffset);
  } else {
    return Status::ERROR();
  }
  return Status::OK();
}

Status CPUInterp::onResize(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) {
  if (mResizeType != 2) {
    return Status::OK();
  }
  const int inW = inputs[0]->buffer().dim[3].extent;
  const int inH = inputs[0]->buffer().dim[2].extent;
  const int outW = outputs[0]->buffer().dim[3].extent;
  const int outH = outputs[0]->buffer().dim[2].extent;
  const float xScaling = mWidthScale;
  const float yScaling = mHeightScale;

  mWidthPosition.buffer().dim[0].extent = 2 * outW;
  mWidthPosition.buffer().dimensions = 1;
  mWidthPosition.setType(DataType_DT_INT32);

  mWidthFactor.buffer().dim[0].extent = outW;
  mWidthFactor.buffer().dimensions = 1;
  mWidthFactor.setType(DataType_DT_FLOAT);

  mHeightPosition.buffer().dim[0].extent = 2 * outH;
  mHeightPosition.buffer().dimensions = 1;
  mHeightPosition.setType(DataType_DT_INT32);

  mHeightFactor.buffer().dim[0].extent = outH;
  mHeightFactor.buffer().dimensions = 1;
  mHeightFactor.setType(DataType_DT_FLOAT);
  bool res = backend()->onAcquireBuffer(&mWidthPosition, Device::STATIC);
  res = res && backend()->onAcquireBuffer(&mWidthFactor, Device::STATIC);
  res = res && backend()->onAcquireBuffer(&mHeightPosition, Device::STATIC);
  res = res && backend()->onAcquireBuffer(&mHeightFactor, Device::STATIC);
  if (!res) {
    return Status::ERROR();
  }
  auto _wPosition = mWidthPosition.host<int>();
  auto _wFactor = mWidthFactor.host<float>();

  // Compute Line Position
  for (int x = 0; x < outW; ++x) {
    float srcX = x * xScaling + mWidthOffset;
    int x1 = floor(srcX);
    float x2Factor = srcX - x1;

    _wFactor[x] = x2Factor;
    _wPosition[2 * x + 0] = CLAMP(x1, 0, inW - 1);
    _wPosition[2 * x + 1] = CLAMP(x1 + 1, 0, inW - 1);
  }

  auto _hPosition = mHeightPosition.host<int>();
  auto _hFactor = mHeightFactor.host<float>();

  for (int y = 0; y < outH; ++y) {
    float srcY = y * yScaling + mHeightOffset;
    int y1 = floor(srcY);
    float y2Factor = srcY - y1;

    _hFactor[y] = y2Factor;
    _hPosition[2 * y + 0] = CLAMP(y1, 0, inH - 1);
    _hPosition[2 * y + 1] = CLAMP(y1 + 1, 0, inH - 1);
  }

  int threadNumber = ((CPUDevice *)backend())->threadNumber();

  mLineBuffer.buffer().dim[0].extent = 2 * 4 * outW * threadNumber;
  mLineBuffer.buffer().dimensions = 1;
  mLineBuffer.setType(DataType_DT_FLOAT);
  res = backend()->onAcquireBuffer(&mLineBuffer, Device::DYNAMIC);
  if (!res) {
    return Status::ERROR();
  }
  backend()->onReleaseBuffer(&mLineBuffer, Device::DYNAMIC);

  return Status::OK();
}

class CPUInterpCreator : public CPUDevice::Creator {
 public:
  virtual Operation *onCreate(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs,
                              const tars::Op *op, Device *backend) const {
    auto interp = op->main_as_Interp();
    return new CPUInterp(backend, interp->resizeType(), interp->widthScale(),
                         interp->heightScale(), interp->widthOffset(),
                         interp->heightOffset());
  }
};
REGISTER_CPU_OP_CREATOR(CPUInterpCreator, OpType_Interp);

}  // namespace tars

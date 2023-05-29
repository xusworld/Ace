//
//  CPUInterp.cpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>

#include "core/TensorUtils.hpp"
#include "device/cpu/CPUDevice.h"
#include "device/cpu/CPUInterp3D.hpp"
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

CPUInterp3D::CPUInterp3D(Device *backend, int resizeType, float widthScale,
                         float heightScale, float depthScale, float widthOffset,
                         float heightOffset, float depthOffset)
    : CPUResizeCommon(backend),
      mResizeType(resizeType),
      mWidthScale(widthScale),
      mHeightScale(heightScale),
      mDepthScale(depthScale),
      mWidthOffset(widthOffset),
      mHeightOffset(heightOffset),
      mDepthOffset(depthOffset) {
  // nothing to do
}

CPUInterp3D::~CPUInterp3D() {
  if (mInit && mResizeType == 2) {
    backend()->onReleaseBuffer(&mWidthPosition, Device::STATIC);
    backend()->onReleaseBuffer(&mWidthFactor, Device::STATIC);
    backend()->onReleaseBuffer(&mHeightPosition, Device::STATIC);
    backend()->onReleaseBuffer(&mHeightFactor, Device::STATIC);
    backend()->onReleaseBuffer(&mDepthPosition, Device::STATIC);
    backend()->onReleaseBuffer(&mDepthFactor, Device::STATIC);
  }
}
// TODO: wtd interp3d
Status CPUInterp3D::onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) {
  auto &input = inputs[0]->buffer();
  auto &output = outputs[0]->buffer();

  if (mResizeType == 1) {
    // Nearstneighbor
    CPUResizeNearestneighbor3DC4(input, output, mWidthScale, mHeightScale,
                                 mDepthScale, mWidthOffset, mHeightOffset,
                                 mDepthOffset);
  } else if (mResizeType == 2) {
    // bilinear
    // CPUResizeBilinearC4(input, output, mWidthPosition.host<int>(),
    // mWidthFactor.host<float>(),
    //                    mHeightPosition.host<int>(),
    //                    mHeightFactor.host<float>(),
    //                    mLineBuffer.host<float>(),
    //                    ((CPUDevice *)backend())->threadNumber());
    MNN_ERROR(
        "Bilinear interpolation is not implemented in interp3D. Do nothing...");
  } else if (mResizeType == 3) {
    // cubic
    // CPUResizeCubicC4(input, output, mWidthScale, mHeightScale, mWidthOffset,
    // mHeightOffset);
    MNN_ERROR(
        "cubic interpolation is not implemented in interp3D. Do nothing...");
  } else if (mResizeType == 4) {
    // Nearstneighbor
    CPUResizeNearestneighbor3DRoundC4(input, output, mWidthScale, mHeightScale,
                                      mWidthOffset, mHeightOffset);
  } else {
    return Status::ERROR();
  }
  auto outPtr = outputs[0]->host<float>();
  return Status::OK();
}

Status CPUInterp3D::onResize(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) {
  if (mResizeType != 2) {
    return Status::OK();
  }
  const int inW = inputs[0]->buffer().dim[4].extent;
  const int inH = inputs[0]->buffer().dim[3].extent;
  const int inD = inputs[0]->buffer().dim[2].extent;
  const int outW = outputs[0]->buffer().dim[4].extent;
  const int outH = outputs[0]->buffer().dim[3].extent;
  const int outD = outputs[0]->buffer().dim[2].extent;
  const float xScaling = mWidthScale;
  const float yScaling = mHeightScale;
  const float zScaling = mDepthScale;

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

  mDepthPosition.buffer().dim[0].extent = 2 * outD;
  mDepthPosition.buffer().dimensions = 1;
  mDepthPosition.setType(DataType_DT_INT32);

  mDepthFactor.buffer().dim[0].extent = outD;
  mDepthFactor.buffer().dimensions = 1;
  mDepthFactor.setType(DataType_DT_FLOAT);

  bool res = backend()->onAcquireBuffer(&mWidthPosition, Device::STATIC);
  res = res && backend()->onAcquireBuffer(&mWidthFactor, Device::STATIC);
  res = res && backend()->onAcquireBuffer(&mHeightPosition, Device::STATIC);
  res = res && backend()->onAcquireBuffer(&mHeightFactor, Device::STATIC);
  res = res && backend()->onAcquireBuffer(&mDepthPosition, Device::STATIC);
  res = res && backend()->onAcquireBuffer(&mDepthFactor, Device::STATIC);
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

  auto _dPosition = mDepthPosition.host<int>();
  auto _dFactor = mDepthFactor.host<float>();

  for (int z = 0; z < outD; ++z) {
    float srcZ = z * zScaling + mDepthOffset;
    int z1 = floor(srcZ);
    float z2Factor = srcZ - z1;

    _dFactor[z] = z2Factor;
    _dPosition[2 * z + 0] = CLAMP(z1, 0, inD - 1);
    _dPosition[2 * z + 1] = CLAMP(z1 + 1, 0, inD - 1);
  }

  int threadNumber = ((CPUDevice *)backend())->threadNumber();
  // TODO line buffer??
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

class CPUInterp3DCreator : public CPUDevice::Creator {
 public:
  virtual Operation *onCreate(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs,
                              const tars::Op *op, Device *backend) const {
    auto interp3D = op->main_as_Interp();
    return new CPUInterp3D(backend, interp3D->resizeType(),
                           interp3D->widthScale(), interp3D->heightScale(),
                           interp3D->depthScale(), interp3D->widthOffset(),
                           interp3D->heightOffset(), interp3D->depthOffset());
  }
};
REGISTER_CPU_OP_CREATOR(CPUInterp3DCreator, OpType_Interp3D);

}  // namespace tars

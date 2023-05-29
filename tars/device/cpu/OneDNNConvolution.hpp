#include "core/operation.h"
#include "device/cpu/CPUDevice.h"
namespace tars {
namespace OneDNN {
Operation *createConvolution(const Convolution2DCommon *common, Device *b,
                             const float *originWeight, size_t originWeightSize,
                             const float *bias, size_t biasSize);
};
};  // namespace tars

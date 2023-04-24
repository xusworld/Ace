#include "backend/cpu/CPUBackend.hpp"
#include "core/Execution.hpp"
namespace ace {
namespace OneDNN {
Execution *createConvolution(const Convolution2DCommon *common, Backend *b,
                             const float *originWeight, size_t originWeightSize,
                             const float *bias, size_t biasSize);
};
};  // namespace ace

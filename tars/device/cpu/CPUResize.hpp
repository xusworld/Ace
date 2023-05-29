//
//  CPUResize.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUResize_hpp
#define CPUResize_hpp

#include "core/AutoStorage.h"
#include "core/operation.h"

namespace tars {

class CPUResizeCommon : public Operation {
 public:
  CPUResizeCommon(Device *backend) : Operation(backend) {
    // Do nothing
  }
  virtual ~CPUResizeCommon() = default;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) = 0;
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) = 0;

  void CPUResizeCubicC4(halide_buffer_t &input, halide_buffer_t &output,
                        float wScale, float hScale, float wOffset,
                        float hOffset);
  void CPUResizeBilinearC4(halide_buffer_t &input, halide_buffer_t &output,
                           const int *widthPosition, const float *widthFactor,
                           const int *heightPosition, const float *heightFactor,
                           float *lineBuffer, int threadNumber);
  void CPUResizeNearestneighborC4(halide_buffer_t &input,
                                  halide_buffer_t &output, float wScale,
                                  float hScale, float wOffset = 0.f,
                                  float hOffset = 0.f);
  void CPUResizeNearestneighborRoundC4(halide_buffer_t &input,
                                       halide_buffer_t &output, float wScale,
                                       float hScale, float wOffset = 0.f,
                                       float hOffset = 0.f);

  void CPUResizeNearestneighbor3DC4(halide_buffer_t &input,
                                    halide_buffer_t &output, float wScale,
                                    float hScale, float dScale,
                                    float wOffset = 0.f, float hOffset = 0.f,
                                    float dOffset = 0.f);
  void CPUResizeNearestneighbor3DRoundC4(halide_buffer_t &input,
                                         halide_buffer_t &output, float wScale,
                                         float hScale, float dScale,
                                         float wOffset = 0.f,
                                         float hOffset = 0.f,
                                         float dOffset = 0.f);
};

}  // namespace tars

#endif /* CPUResize_hpp */

#pragma once

#include <cassert>

// Copy From weihuali
namespace kernels {
namespace cpu {
namespace jit {

struct DotProduct {
  virtual ~DotProduct() = default;

  // From benchmark result, parameter n should not be fixed!
  static DotProduct* Create(bool fix_param_n = false);

  // optional
  virtual void FixParams(int n) { return; }

  virtual float operator()(float* x, float* y, int n) {
    assert(false && "not implement!");
    return 0;
  }
};

}  // namespace jit
}  // namespace cpu
}  // namespace kernels

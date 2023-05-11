#include "jit_dot_product.h"

#include <cstring>

#include "jit_generator.h"

// Copy From weihuali
namespace kernels {
namespace cpu {
namespace jit {

using Xbyak::Xmm;
using Xbyak::Ymm;

static inline int float2int(float x) {
  int t;
  std::memcpy(&t, &x, sizeof(int));
  return t;
}

constexpr int kPack = 8;
constexpr int kUnroll = 8;

class DotProductImpl : public DotProduct, public JITGenerator {
 public:
  void FixParams(int n) override {
    n_ = n / kPack;
    fix_param_n_ = true;
    this->CreateKernel();
  }

  float operator()(float *x, float *y, int n) override {
    assert(jit_kernel_ && "Forget to set param n");

    float result;

    n /= kPack;

    // unrolled loop number
    int m = n / kUnroll;

    // call jit kernel
    JITGenerator::operator()(x, y, n, &result, m);

    return result;
  }

 private:
  void GenerateCode() override;

 private:
  bool fix_param_n_ = false;
  int n_;
  const Xbyak::Reg64 &reg_x_ = rdi;
  const Xbyak::Reg64 &reg_y_ = rsi;
  const Xbyak::Reg64 &reg_n_ = rdx;
  const Xbyak::Reg64 &reg_output_ = rcx;
  const Xbyak::Reg64 &reg_m_ = r8;

  Xbyak::Reg64 reg_tmp_ = r9;
  Xbyak::Reg64 reg_loop_ = r10;

  Ymm px_ = Ymm(14);
  Ymm py_ = Ymm(15);
};

// DotProduct(float* x, float* y, int n, float* output, int m);
//              rdi,         rsi,   rdx,       rcx,   r8
void DotProductImpl::GenerateCode() {
  Preamble();
  uni_vzeroupper();

  constexpr int64_t kStep = sizeof(float) * kUnroll * kPack;

  for (int k = 0; k < kUnroll; ++k) {
    vpxor(Ymm(k), Ymm(k), Ymm(k));
  }

  if (fix_param_n_) {
    for (int i = 0; i < n_ / kUnroll; ++i) {
      for (int k = 0; k < kUnroll; ++k) {
        vmovaps(px_, ptr[reg_x_ + k * sizeof(float) * kPack]);
        vfmadd231ps(Ymm(k), px_, ptr[reg_y_ + k * sizeof(float) * kPack]);
      }
      add(reg_x_, kStep);
      add(reg_y_, kStep);
    }
    // handle left
    int nleft = n_ - (n_ / kUnroll) * kUnroll;
    for (int k = 0; k < nleft; ++k) {
      vmovaps(px_, ptr[reg_x_ + k * sizeof(float) * kPack]);
      vfmadd231ps(Ymm(k), px_, ptr[reg_y_ + k * sizeof(float) * kPack]);
    }
  } else {
    Xbyak::Label unroll_loop, unroll_end, left_loop, left_end;

    // skip if m <= 0
    // so we can save one jump instruction
    // ref: Agner "Optimizing subroutines in assembly language" Example 12.1c.
    test(reg_m_, reg_m_);
    jng(unroll_end, T_NEAR);

    // experiment
    // ref: Agner "Optimizing subroutines in assembly language" Example 12.6c.
#define USE_NAIVE_METHOD 1
#if USE_NAIVE_METHOD
    mov(reg_loop_, reg_m_);
#else
    // move reg_x_, reg_y_ to end of unrolled part of array
    imul(reg_loop_, reg_m_, kStep);
    add(reg_x_, reg_loop_);
    add(reg_y_, reg_loop_);
    neg(reg_loop_);
#endif
    L(unroll_loop);
    {
      // unrool loop
      for (int k = 0; k < kUnroll; ++k) {
#if USE_NAIVE_METHOD
        vmovaps(px_, ptr[reg_x_ + k * sizeof(float) * kPack]);
        vfmadd231ps(Ymm(k), px_, ptr[reg_y_ + k * sizeof(float) * kPack]);
#else
        vmovaps(px_, ptr[reg_x_ + reg_loop_ + k * sizeof(float) * kPack]);
        vfmadd231ps(Ymm(k), px_,
                    ptr[reg_y_ + reg_loop_ + k * sizeof(float) * kPack]);
#endif
      }
#if USE_NAIVE_METHOD
      add(reg_x_, kStep);
      add(reg_y_, kStep);
      sub(reg_loop_, 1);
      jnz(unroll_loop);
#else
      add(reg_loop_, kStep);  // sizeof(float) * kPack * kUnroll
      js(unroll_loop);
#endif
    }
    L(unroll_end);

#undef USE_NAIVE_METHOD

    // left loop
    // Best use padded data for performance!
    imul(reg_m_, reg_m_, kUnroll);
    sub(reg_n_, reg_m_);

    // when no left
    test(reg_n_, reg_n_);
    jng(left_end, T_NEAR);

    L(left_loop);
    {
      vmovaps(px_, ptr[reg_x_]);
      vmovaps(py_, ptr[reg_y_]);
      // How to use different ymmm register?
      vfmadd231ps(Ymm(0), px_, py_);
      add(reg_x_, sizeof(float) * kPack);
      add(reg_y_, sizeof(float) * kPack);

      sub(reg_n_, 1);
      jnz(left_loop);
    }
    L(left_end);
  }

  int m = kUnroll;
  while (m > 1) {
    m /= 2;
    for (int k = 0; k < m; ++k) {
      vaddps(Ymm(k), Ymm(k), Ymm(k + m));
    }
  }

  vextractf128(xmm1, ymm0, 1);
  vaddps(xmm0, xmm1);
  vhaddps(xmm0, xmm0);
  vhaddps(xmm0, xmm0);
  vmovss(ptr[rcx], xmm0);

  Postamble();
}

DotProduct *DotProduct::Create(bool fix_param_n) {
  auto *impl = new DotProductImpl();
  if (!fix_param_n) impl->CreateKernel();
  return impl;
}

}  // namespace jit
}  // namespace cpu
}  // namespace kernels

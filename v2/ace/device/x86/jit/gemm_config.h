#pragma once

namespace kernels {
namespace cpu {
namespace jit {

template <typename a_t, typename b_t, typename c_t>
struct gemm_config {
  typedef void (*fetch_n_func_t)(const int m, const a_t* a, const int lda,
                                 a_t* b, const int ldb, const int block_size);
  typedef void (*fetch_t_func_t)(const int m, const a_t* a, const int lda,
                                 a_t* b, const int ldb, const int block_size);
  typedef void (*sgemm_ker_func_t)(const int K, const float* alpha_ptr,
                                   const a_t* src_a, int lda, const b_t* src_b,
                                   int ldb, const float* beta_ptr, c_t* dst,
                                   int ldc);

  gemm_config(const char* trans_a, const char* trans_b, const int m,
              const int n, const int k, const float* alpha, const a_t* a,
              const int lda, const b_t* b, const int ldb, const float* beta,
              c_t* c, const int ldc, const int m_block = 16,
              const int n_block = 6);

  const a_t* a_;
  const b_t* b_;
  c_t* c_;

  const int m_, n_, k_;
  const int lda_, ldb_, ldc_;

  const float* alpha_;
  const float* beta_;

  // block size for data packing
  int m_block_;
  int n_block_;

  // block size for kernel register blocking
  int kernel_m_r_;
  int kernel_n_r_;

  // block size for matrix spliting
  int M_c_;
  int K_c_;

  constexpr static int nb_kernels_m = 16;
  constexpr static int nb_kernels_n = 6;

  fetch_t_func_t pack_t_ker_[nb_kernels_m + 1];
  fetch_t_func_t pack_t_4x16_ker_;
  fetch_t_func_t pack_t_4x8_ker_;

  fetch_n_func_t pack_n_ker_[nb_kernels_n + 1];

  sgemm_ker_func_t kernels_[nb_kernels_m + 1][nb_kernels_n + 1];

 private:
  void init_jit_kernel();
};

}  // namespace jit
}  // namespace cpu
}  // namespace kernels
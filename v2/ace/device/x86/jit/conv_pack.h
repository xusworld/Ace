#pragma once

#include "config.h"

namespace kernels {
namespace cpu {
namespace jit {

template <typename T>
void pack_col_a_n(const T *a, int lda, T *b, int ldb, int m, int n,
                  conv_gemm_config<T, T, T> &conv_gemm_conf);

template <typename T>
void pack_col_b_n(const T *a, int lda, T *b, int ldb, int m, int n,
                  conv_gemm_config<T, T, T> &conv_gemm_conf);

template <typename T>
void pack_col_a_t(const T *a, int lda, T *b, int ldb, int m, int n,
                  conv_gemm_config<T, T, T> &conv_gemm_conf);

}  // namespace jit
}  // namespace cpu
}  // namespace kernels
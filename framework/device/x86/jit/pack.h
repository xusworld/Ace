#pragma once

#include "config.h"
#include "gemm_config.h"

namespace kernels {
namespace cpu {
namespace jit {

template <typename T>
void pack_t(const T *a, int lda, T *b, int ldb, int m, int n,
            gemm_config<T, T, T> &gemm_conf);

template <typename T>
void pack_n(const T *a, int lda, T *b, int ldb, int m, int n,
            gemm_config<T, T, T> &gemm_conf);

}  // namespace jit
}  // namespace cpu
}  // namespace kernels
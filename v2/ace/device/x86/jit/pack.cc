#include <algorithm>
#include <iostream>

#include "config.h"
#include "conv_pack.h"
#include "cpu/utils.h"
#include "pack.h"

namespace kernels {
namespace cpu {
namespace jit {

//  pack block_size on non-leading dimension, n denotes no-transpose.
//  eg. input:   A MxN matrix in col major, so the storage-format is (N, M)
//      output:  B MxN matrix in col major(N-packed), so the storage-format is
//                    (divUp(N, block_size), M, block_size)
template <typename T>
void pack_n(const T *a, int lda, T *b, int ldb, int m, int n,
            gemm_config<T, T, T> &gemm_conf) {
  // block_size = 6
  int block_size = gemm_conf.n_block_;
  int i = 0;

  for (; n - i >= block_size; i += block_size) {
    const T *cur_a = a + i * lda;
    T *cur_b = b + i * ldb;
    gemm_conf.pack_n_ker_[block_size](m, cur_a, lda, cur_b, ldb, block_size);
  }

  int tail = n - i;
  if (tail > 0) {
    const T *cur_a = a + i * lda;
    T *cur_b = b + i * ldb;
    gemm_conf.pack_n_ker_[tail](m, cur_a, lda, cur_b, ldb, block_size);
  }
}

template void pack_n<float>(const float *a, int lda, float *b, int ldb, int m,
                            int n, gemm_config<float, float, float> &gemm_conf);

//  pack block_size on leading dimension, t denotes transpose.
//  eg. input:   A MxN matrix in row major, so the storage-format is (M, N)
//      output:  B MxN matrix in col major(N-packed), so the storage-format is
//                    (divUp(N, block_size), M, block_size)
template <typename T>
void pack_t(const T *a, int lda, T *b, int ldb, int m, int n,
            gemm_config<T, T, T> &gemm_conf) {
  int block_size = gemm_conf.m_block_;
  int i = 0;
  if (block_size == 16) {
    for (; i + 64 <= n; i += 64) {
      const T *cur_a = a + i;
      T *cur_b = b + i * ldb;
      gemm_conf.pack_t_4x16_ker_(m, cur_a, lda, cur_b, ldb, block_size);
    }
  } else if (block_size == 8) {
  }

  for (; i < n;) {
    const T *cur_a = a + i;
    T *cur_b = b + DivDown(i, block_size) * ldb + i % block_size;
    int cur_n = std::min(n - i, block_size);
    switch (cur_n) {
      case 1:
        gemm_conf.pack_t_ker_[1](m, cur_a, lda, cur_b, ldb, block_size);
        i += 1;
        break;
      case 2:
        gemm_conf.pack_t_ker_[2](m, cur_a, lda, cur_b, ldb, block_size);
        i += 2;
        break;
      case 3:
        gemm_conf.pack_t_ker_[3](m, cur_a, lda, cur_b, ldb, block_size);
        i += 3;
        break;
      case 4:
      case 5:
      case 6:
      case 7:
        gemm_conf.pack_t_ker_[4](m, cur_a, lda, cur_b, ldb, block_size);
        i += 4;
        break;
      case 8:
      case 9:
      case 10:
      case 11:
      case 12:
      case 13:
      case 14:
      case 15:
        gemm_conf.pack_t_ker_[8](m, cur_a, lda, cur_b, ldb, block_size);
        i += 8;
        break;
      default:
        gemm_conf.pack_t_ker_[16](m, cur_a, lda, cur_b, ldb, block_size);
        i += 16;
        break;
    }
  }
  /*
  if (block_n == 16) {

      int i=0;
      // must be a perfect loop, n % block_n = 0
      for(;i + 64 <=n;i+=64) {
          const float * cur_a = a + i;
          float * cur_b = b + i * ldb;
          sgemm_fetch_t_4x16(m, cur_a, lda, cur_b, ldb, 16);
      }

  }

  */
}

template void pack_t<float>(const float *a, int lda, float *b, int ldb, int m,
                            int n, gemm_config<float, float, float> &gemm_conf);

}  // namespace jit
}  // namespace cpu
}  // namespace kernels

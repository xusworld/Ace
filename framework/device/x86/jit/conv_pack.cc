#include <immintrin.h>
#include <stdio.h>
#include <xmmintrin.h>

#include <algorithm>

#include "config.h"
#include "cpu/utils.h"
#include "pack.h"

namespace kernels {
namespace cpu {
namespace jit {

//  pack block_size on non-leading dimension, n denotes no-transpose.
//  eg. input:   A MxN matrix in col major, so the storage-format is (N, M)
//      output:  B MxN matrix in col major(N-packed), so the storage-format is
//                    (DivUp(N, block_size), M, block_size)
template <typename T>
void pack_col_b_n(const T *a, int lda, T *b, int ldb, int m, int n,
                  conv_gemm_config<T, T, T> &conv_gemm_conf) {
  int block_size = conv_gemm_conf.n_block_;
  int i = 0;
  for (; n - i >= block_size; i += block_size) {
    const T *cur_a = a + i * lda;
    T *cur_b = b + i * ldb;
    conv_gemm_conf.pack_n_ker_[block_size](m, cur_a, lda, cur_b, ldb,
                                           block_size);
  }
  int tail = n - i;
  if (tail > 0) {
    const T *cur_a = a + i * lda;
    T *cur_b = b + i * ldb;
    conv_gemm_conf.pack_n_ker_[tail](m, cur_a, lda, cur_b, ldb, block_size);
  }
}

template void pack_col_b_n<float>(
    const float *a, int lda, float *b, int ldb, int m, int n,
    conv_gemm_config<float, float, float> &conv_gemm_conf);

//  pack block_size on leading dimension, t denotes transpose.
//  eg. input:   A MxN matrix in row major, so the storage-format is (M, N)
//      output:  B MxN matrix in col major(N-packed), so the storage-format is
//                    (DivUp(N, block_size), M, block_size)
template <typename T>
void pack_col_a_n(const T *a, int lda, T *b, int ldb, int m, int n,
                  conv_gemm_config<T, T, T> &conv_gemm_conf) {
  int block_size = conv_gemm_conf.m_block_;
  int i = 0;

  LOG(INFO) << "block_size: " << block_size;
  if (block_size == 16) {
    for (; i + 64 <= n; i += 64) {
      const T *cur_a = a + i;
      T *cur_b = b + i * ldb;
      LOG(INFO) << "pack_t_4x16_ker_ "
                << "conv_gemm_conf.pack_t_4x16_ker_: "
                << conv_gemm_conf.pack_t_4x16_ker_;
      LOG(INFO) << "m: " << m << " cur_a: " << cur_a << " lda: " << lda
                << " cur_b: " << cur_b << " ldb: " << ldb
                << " block_size: " << block_size;

      conv_gemm_conf.pack_t_4x16_ker_(m, cur_a, lda, cur_b, ldb, block_size);
    }
  } else if (block_size == 8) {
  }

  LOG(INFO) << "n: " << n;
  for (; i < n;) {
    LOG(INFO) << "i: " << i;
    const T *cur_a = a + i;
    T *cur_b = b + DivDown(i, block_size) * ldb + i % block_size;
    int cur_n = std::min(n - i, block_size);
    switch (cur_n) {
      case 1:
        conv_gemm_conf.pack_t_ker_[1](m, cur_a, lda, cur_b, ldb, block_size);
        i += 1;
        break;
      case 2:
        conv_gemm_conf.pack_t_ker_[2](m, cur_a, lda, cur_b, ldb, block_size);
        i += 2;
        break;
      case 3:
        conv_gemm_conf.pack_t_ker_[3](m, cur_a, lda, cur_b, ldb, block_size);
        i += 3;
        break;
      case 4:
      case 5:
      case 6:
      case 7:
        conv_gemm_conf.pack_t_ker_[4](m, cur_a, lda, cur_b, ldb, block_size);
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
        conv_gemm_conf.pack_t_ker_[8](m, cur_a, lda, cur_b, ldb, block_size);
        i += 8;
        break;
      default:
        conv_gemm_conf.pack_t_ker_[16](m, cur_a, lda, cur_b, ldb, block_size);
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

template void pack_col_a_n<float>(
    const float *a, int lda, float *b, int ldb, int m, int n,
    conv_gemm_config<float, float, float> &conv_gemm_conf);

template <int blk, typename T>
void pack_a_t_trans(const T *src, int lda, T *dst, int cur_k, int block_size) {
  for (int j = 0; j < blk; j++) {
    auto src_j = src + j * lda;
    auto dst_j = dst + j;
    for (int k = 0; k < cur_k; k++) {
      dst_j[k * block_size] = src_j[k];
    }
  }
}

// lda -> total_k
// ldb -> m_block_size (M_c)
// cur_k
// cur_m
template <typename T>
void pack_col_a_t(const T *src_a, int lda, T *src_b, int ldb, int cur_k,
                  int cur_m, conv_gemm_config<T, T, T> &conv_gemm_conf) {
  int block_size = conv_gemm_conf.m_block_;
  int i = 0;

  if (block_size == 16) {
    for (; i + 15 < cur_m; i += 16) {
      auto a_ptr = src_a + i * lda;
      auto b_ptr = src_b + i * ldb;
      pack_a_t_trans<16, T>(a_ptr, lda, b_ptr, cur_k, block_size);
    }
  }
  for (; i + 7 < cur_m; i += 8) {
    auto a_ptr = src_a + i * lda;
    auto b_ptr = src_b + DivDown(i, block_size) * ldb + i % block_size;
    pack_a_t_trans<8, T>(a_ptr, lda, b_ptr, cur_k, block_size);
  }
  for (; i + 3 < cur_m; i += 4) {
    auto a_ptr = src_a + i * lda;
    auto b_ptr = src_b + DivDown(i, block_size) * ldb + i % block_size;
    pack_a_t_trans<4, T>(a_ptr, lda, b_ptr, cur_k, block_size);
  }
  for (; i + 2 < cur_m; i += 3) {
    auto a_ptr = src_a + i * lda;
    auto b_ptr = src_b + DivDown(i, block_size) * ldb + i % block_size;
    pack_a_t_trans<3, T>(a_ptr, lda, b_ptr, cur_k, block_size);
  }
  for (; i + 1 < cur_m; i += 2) {
    auto a_ptr = src_a + i * lda;
    auto b_ptr = src_b + DivDown(i, block_size) * ldb + i % block_size;
    pack_a_t_trans<2, T>(a_ptr, lda, b_ptr, cur_k, block_size);
  }
  for (; i < cur_m; i++) {
    auto a_ptr = src_a + i * lda;
    auto b_ptr = src_b + DivDown(i, block_size) * ldb + i % block_size;
    pack_a_t_trans<1, T>(a_ptr, lda, b_ptr, cur_k, block_size);
  }
}
template void pack_col_a_t<float>(
    const float *a, int lda, float *b, int ldb, int m, int n,
    conv_gemm_config<float, float, float> &conv_gemm_conf);

}  // namespace jit
}  // namespace cpu
}  // namespace kernels

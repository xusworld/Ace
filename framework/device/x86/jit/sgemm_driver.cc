#include <mm_malloc.h>

#include <algorithm>
#include <cmath>

#include "conv_pack.h"
#include "cpu/omp_utils.h"
#include "cpu/utils.h"
#include "gemm_driver.h"
#include "pack.h"

namespace kernels {
namespace cpu {
namespace jit {

using dim_t = long int;

void sgemm_block_n(dim_t M, dim_t N, dim_t K, const float alpha,
                   const float *src_a, dim_t lda, const float *src_b, dim_t ldb,
                   const float beta, float *dst, dim_t ldc,
                   gemm_config<float, float, float> &gemm_conf) {
  dim_t K_c = gemm_conf.K_c_;
  dim_t m_block = gemm_conf.m_block_;

  for (dim_t i = 0; i < M;) {
    dim_t cur_m = std::min(int(M - i), gemm_conf.kernel_m_r_);

    const float *cur_a = src_a + DivDown(i, m_block) * K_c + i % m_block;
    const float *cur_b = src_b;
    float *cur_c = dst + i;

    switch (cur_m) {
      case 1:
        gemm_conf.kernels_[1][N](K, &alpha, cur_a, lda, cur_b, ldb, &beta,
                                 cur_c, ldc);
        i += 1;
        break;
      case 2:
      case 3:
        gemm_conf.kernels_[2][N](K, &alpha, cur_a, lda, cur_b, ldb, &beta,
                                 cur_c, ldc);
        i += 2;
        break;
      case 4:
      case 5:
      case 6:
      case 7:
        gemm_conf.kernels_[4][N](K, &alpha, cur_a, lda, cur_b, ldb, &beta,
                                 cur_c, ldc);
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
        gemm_conf.kernels_[8][N](K, &alpha, cur_a, lda, cur_b, ldb, &beta,
                                 cur_c, ldc);
        i += 8;
        break;
      default:
        gemm_conf.kernels_[16][N](K, &alpha, cur_a, lda, cur_b, ldb, &beta,
                                  cur_c, ldc);
        i += 16;
        break;
    }
  }
}

void sgemm_nn_col_major(dim_t M, dim_t N, dim_t K, const float alpha,
                        const float *src_a, dim_t lda, const float *src_b,
                        dim_t ldb, const float beta, float *dst, dim_t ldc) {
  if (alpha == 0) {
    return;
  }
#ifdef DEBUG_MEM
  LOG(INFO) << "(M, N, K): "
            << "( " << M << ", " << N << ", " << K << ")";
  LOG(INFO) << "(lda, ldb, ldc): "
            << "( " << lda << ", " << ldb << ", " << ldc << ")";
#endif

  float beta_div_alpha = beta / alpha;
  gemm_config<float, float, float> gemm_conf("N", "N", M, N, K, &alpha, src_b,
                                             lda, src_b, ldb, &beta_div_alpha,
                                             dst, ldc);

  dim_t M_c = gemm_conf.M_c_;
  dim_t K_c = gemm_conf.K_c_;
  dim_t m_block = gemm_conf.m_block_;
  dim_t n_block = gemm_conf.n_block_;

#ifdef DEBUG_MEM
  LOG(INFO) << "(K_c, DivUp(N, n_block): "
            << "( " << K_c << " , " << DivUp(N, n_block) << ")";
#endif
  float *pack_a = (float *)_mm_malloc(M_c * K_c * 4, 32);
  float *pack_b = (float *)_mm_malloc(K_c * DivUp(N, n_block) * 4, 32);

  dim_t i, j, k;
  i = j = k = 0;

#ifdef DEBUG_MEM
  LOG(INFO) << "(M_c, K_c): "
            << "(" << M_c << ", " << K_c << ")";
  LOG(INFO) << "(m_block, n_block): "
            << "(" << m_block << ", " << n_block << ")";
  LOG(INFO) << "Outer Loop times: " << K / K_c
            << " , Inner Loop times: " << M / M_c;
#endif

  for (k = 0; k < K; k += K_c) {
    float cur_beta = 1.0 / alpha;
    if (k == 0) cur_beta = beta_div_alpha;

    dim_t cur_k = std::min(K - k, K_c);

    // pack b -> K_c * N;

#ifdef DEBUG_MEM
    LOG(INFO) << "(src_b + k, ldb, pack_b, K_c, cur_k, N)" << src_b + k
              << " , ldb = " << ldb << " , pack_b = " << pack_b
              << " , K_c  = " << K_c << " , cur_k = " << cur_k
              << " , N = " << N;

#endif

    pack_n(src_b + k, ldb, pack_b, K_c, cur_k, N, gemm_conf);

    for (i = 0; i < M; i += M_c) {
      // LOG(INFO) << "i = " << i;
      dim_t cur_m = std::min(M - i, M_c);
      // pack a -> M_c * K_c;
      pack_t(src_a + i + k * lda, lda, pack_a, K_c, cur_k, cur_m, gemm_conf);

      // 计算 (64, N) x (N, 6) = (64, 6)
      for (j = 0; j < N;) {
        dim_t cur_n = std::min(int(N - j), gemm_conf.kernel_n_r_);
        float *cur_c = dst + i + j * ldc;

        const float *packed_cur_b =
            pack_b + DivDown(j, n_block) * K_c + j % n_block;
        // Inner Kernel
#ifdef DEBUG_MEM
        LOG(INFO) << "Index: " << j << "(cur_m, cur_n, cur_k)"
                  << "(" << cur_m << " , " << cur_n << " , " << cur_k << ")";

#endif
        // 计算 (64, 8) x (8, 6) = (64, 6)
        sgemm_block_n(cur_m, cur_n, cur_k, alpha, pack_a, lda, packed_cur_b,
                      ldb, cur_beta, cur_c, ldc, gemm_conf);
        j += cur_n;
      }
    }
  }

  _mm_free(pack_a);
  _mm_free(pack_b);
}

/*
void sgemm_col_naive(OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                     OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, dim_t M,
                     dim_t N, dim_t K, const float alpha, const float *src_a,
                     dim_t lda, const float *src_b, dim_t ldb, const float beta,
                     float *dst, dim_t ldc) {
  dim_t a_stride_m = TransA == CblasNoTrans ? 1 : lda;
  dim_t a_stride_k = TransA == CblasNoTrans ? lda : 1;
  dim_t b_stride_k = TransB == CblasNoTrans ? 1 : ldb;
  dim_t b_stride_n = TransB == CblasNoTrans ? ldb : 1;

  for (dim_t m = 0; m < M; m++) {
    for (dim_t n = 0; n < N; n++) {
      float acc = 0.f;
      const float *a_ptr = src_a + m * a_stride_m;
      const float *b_ptr = src_b + n * b_stride_n;

      for (dim_t k = 0; k < K; k++) {
        acc += a_ptr[0] * b_ptr[0];
        a_ptr += a_stride_k;
        b_ptr += b_stride_k;
      }

      dst[m + n * ldc] = alpha * acc + beta * dst[m + n * ldc];
    }
  }
}

void sgemm_row_naive(OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                     OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, dim_t M,
                     dim_t N, dim_t K, const float alpha, const float *src_a,
                     dim_t lda, const float *src_b, dim_t ldb, const float beta,
                     float *dst, dim_t ldc) {
  dim_t a_stride_m = TransA == CblasNoTrans ? lda : 1;
  dim_t a_stride_k = TransA == CblasNoTrans ? 1 : lda;
  dim_t b_stride_k = TransB == CblasNoTrans ? ldb : 1;
  dim_t b_stride_n = TransB == CblasNoTrans ? 1 : ldb;

  for (dim_t m = 0; m < M; m++) {
    for (dim_t n = 0; n < N; n++) {
      float acc = 0.f;
      const float *a_ptr = src_a + m * a_stride_m;
      const float *b_ptr = src_b + n * b_stride_n;

      for (dim_t k = 0; k < K; k++) {
        acc += a_ptr[0] * b_ptr[0];
        a_ptr += a_stride_k;
        b_ptr += b_stride_k;
      }

      dst[m * ldc + n] = alpha * acc + beta * dst[m * ldc + n];
    }
  }
}


void cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order,
                 OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
                 OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha,
                 OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda,
                 OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb,
                 OPENBLAS_CONST float beta, float *C,
                 OPENBLAS_CONST blasint ldc) {
  if (Order == CblasColMajor && TransA == CblasNoTrans &&
      TransB == CblasNoTrans) {
    TNN_NS::sgemm_nn_col_major(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (Order == CblasColMajor) {
    TNN_NS::sgemm_col_naive(TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                            beta, C, ldc);
  } else {
    TNN_NS::sgemm_row_naive(TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                            beta, C, ldc);
  }


  return;
}

int sgemm_opt1(const char *transa, const char *transb, FINTEGER *m, FINTEGER *n,
               FINTEGER *k, const float *alpha, const float *a, FINTEGER *lda,
               const float *b, FINTEGER *ldb, float *beta, float *c,
               FINTEGER *ldc) {
  TNN_NS::sgemm_nn_col_major(*m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c,
                             *ldc);

  return 0;
}
*/

}  // namespace jit
}  // namespace cpu
}  // namespace kernels

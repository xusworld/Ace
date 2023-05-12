#include "gemm_driver.h"

#include "conv_pack.h"
#include "cpu/omp_utils.h"
#include "cpu/utils.h"
#include "pack.h"

namespace kernels {
namespace cpu {
namespace jit {

void conv_sgemm_block_n(int M, int N, int K, const float *src_a, int lda,
                        const float *src_b, int ldb, float *dst, int ldc,
                        const float *bias, int first, int act_type,
                        conv_gemm_config<float, float, float> &conv_gemm_conf) {
  int K_c = conv_gemm_conf.K_c_;
  int m_block = conv_gemm_conf.m_block_;

  for (int i = 0; i < M;) {
    int cur_m = MIN(M - i, conv_gemm_conf.kernel_m_r_);

    const float *cur_a = src_a + DivDown(i, m_block) * K_c + i % m_block;
    const float *cur_b = src_b;
    float *cur_c = dst + i;

    switch (cur_m) {
      case 1:
        conv_gemm_conf.kernels_[1][N](K, cur_a, lda, cur_b, ldb, cur_c, ldc,
                                      bias, first, act_type);
        i += 1;
        break;
      case 2:
      case 3:
        conv_gemm_conf.kernels_[2][N](K, cur_a, lda, cur_b, ldb, cur_c, ldc,
                                      bias, first, act_type);
        i += 2;
        break;
      case 4:
      case 5:
      case 6:
      case 7:
        conv_gemm_conf.kernels_[4][N](K, cur_a, lda, cur_b, ldb, cur_c, ldc,
                                      bias, first, act_type);
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
        conv_gemm_conf.kernels_[8][N](K, cur_a, lda, cur_b, ldb, cur_c, ldc,
                                      bias, first, act_type);
        i += 8;
        break;
      default:
        conv_gemm_conf.kernels_[16][N](K, cur_a, lda, cur_b, ldb, cur_c, ldc,
                                       bias, first, act_type);
        i += 16;
        break;
    }
  }
}

// sgemm col_major a no_trans, b no_trans
// src_a: M * K, lda = M
// src_b: K * N, ldb = K
// dst  : M * N, ldc = M
void conv_sgemm_nn_col_major(
    int M, int N, int K, const float *src_a, int lda, const float *src_b,
    int ldb, float *dst, int ldc, const float *bias, int act_type,
    float *pack_buf, conv_gemm_config<float, float, float> &conv_gemm_conf) {
#ifdef DEBUG_MEM
  LOG(INFO) << "(M, N, K): "
            << "( " << M << ", " << N << ", " << K << ")";
  LOG(INFO) << "(lda, ldb, ldc): "
            << "( " << lda << ", " << ldb << ", " << ldc << ")";
#endif

  int M_c = conv_gemm_conf.M_c_;
  int K_c = conv_gemm_conf.K_c_;
  int m_block = conv_gemm_conf.m_block_;
  int n_block = conv_gemm_conf.n_block_;

  int i, j, k;
  i = j = k = 0;

  int first = 0;
  int post_type;

  auto pack_a_buf = pack_buf;
  auto pack_b_buf =
      pack_buf + DivUp(int(M_c * K_c * sizeof(float)), 32) / sizeof(float);

#ifdef DEBUG_MEM
  LOG(INFO) << "(M_c, K_c): "
            << "( " << M_c << ", " << K_c << ")";
  LOG(INFO) << "(m_block, n_block): "
            << "( " << m_block << ", " << n_block << ")";
#endif

  // if no bias, first set to 1, load c from dst
  if (bias == nullptr) {
    first = 1;
  }

#ifdef DEBUG_MEM
  LOG(INFO) << "Outter Loop count: " << K / K_c;
#endif
  for (k = 0; k < K; k += K_c) {
    if (k + K_c >= K) {
      post_type = act_type;
    } else {
      post_type = 0;
    }

    int cur_k = MIN(K - k, K_c);

    // pack b -> K_c * N;
    // const float *pack_b_k = src_b + k * divUp(N, n_block);
    LOG(INFO) << "pack_col_b_n";
    pack_col_b_n(src_b + k, ldb, pack_b_buf, K_c, cur_k, N, conv_gemm_conf);

#ifdef DEBUG_MEM
    LOG(INFO) << "Inner Loop count: " << M / M_c;
#endif
    for (i = 0; i < M; i += M_c) {
      int cur_m = MIN(M - i, M_c);
      // pack a -> M_c * K_c;
#ifdef DEBUG_MEM
      LOG(INFO) << "pack a --> "
                << "(" << M_c << ", " << K_c << ")";

#endif
      pack_col_a_n(src_a + i + k * lda, lda, pack_a_buf, K_c, cur_k, cur_m,
                   conv_gemm_conf);

      for (j = 0; j < N;) {
        int cur_n = MIN(N - j, conv_gemm_conf.kernel_n_r_);
        float *cur_c = dst + i + j * ldc;

        const float *packed_cur_b =
            pack_b_buf + DivDown(int(j), n_block) * K_c + j % n_block;
        const float *cur_bias = bias + j;
        LOG(INFO) << "conv_sgemm_block_n";
        conv_sgemm_block_n(cur_m, cur_n, cur_k, pack_a_buf, lda, packed_cur_b,
                           ldb, cur_c, ldc, cur_bias, first, post_type,
                           conv_gemm_conf);
        j += cur_n;
      }
    }
    // if k != 0, first = 1
    first = 1;
  }
}

// sgemm col_major a no_trans, b no_trans
// src_a: M * K, lda = M
// src_b: K * N, ldb = K, prepacked
// dst  : M * N, ldc = M
void conv_sgemm_nn_col_major_prepack_b(
    int M, int N, int K, const float *src_a, int lda, const float *src_b,
    int ldb, float *dst, int ldc, const float *bias, int act_type,
    float *src_trans_buf,
    conv_gemm_config<float, float, float> &conv_gemm_conf) {
  int M_c = conv_gemm_conf.M_c_;
  int K_c = conv_gemm_conf.K_c_;
  int m_block = conv_gemm_conf.m_block_;
  int n_block = conv_gemm_conf.n_block_;

  int first = 0;
  int post_type;

  // if no bias, first set to 1, load c from dst
  if (bias == nullptr) {
    first = 1;
  }

  for (int k = 0; k < K; k += K_c) {
    if (k + K_c >= K) {
      post_type = act_type;
    } else {
      post_type = 0;
    }

    int cur_k = MIN(K - k, K_c);

    // pack b -> K_c * N;
    const float *pack_b_k = src_b + k * DivUp(N, n_block);

    OMP_PARALLEL_FOR_DYNAMIC_
    for (int i = 0; i < M; i += M_c) {
      int thread_id = OMP_TID_;
      auto src_trans_per_t = src_trans_buf + thread_id * M_c * K_c;
      int cur_m = MIN(M - i, M_c);
      // pack a -> M_c * K_c;
      pack_col_a_n(src_a + i + k * lda, lda, src_trans_per_t, K_c, cur_k, cur_m,
                   conv_gemm_conf);

      for (int j = 0; j < N;) {
        int cur_n = MIN(N - j, conv_gemm_conf.kernel_n_r_);
        float *cur_c = dst + i + j * ldc;

        const float *packed_cur_b =
            pack_b_k + DivDown(j, n_block) * K_c + j % n_block;
        const float *cur_bias = bias + j;
        conv_sgemm_block_n(cur_m, cur_n, cur_k, src_trans_per_t, lda,
                           packed_cur_b, ldb, cur_c, ldc, cur_bias, first,
                           post_type, conv_gemm_conf);
        j += cur_n;
      }
    }
    // if k != 0, first = 1
    first = 1;
  }
}

// sgemm col_major a trans, b no_trans
// src_a: K * M, lda = K
// src_b: K * N, ldb = K, prepacked
// dst  : M * N, ldc = M
void conv_sgemm_tn_col_major_prepack_b(
    int M, int N, int K, const float *src_a, int lda, const float *src_b,
    int ldb, float *dst, int ldc, const float *bias, int act_type,
    float *src_trans_buf,
    conv_gemm_config<float, float, float> &conv_gemm_conf) {
  int M_c = conv_gemm_conf.M_c_;
  int K_c = conv_gemm_conf.K_c_;
  int m_block = conv_gemm_conf.m_block_;
  int n_block = conv_gemm_conf.n_block_;

  int i, j, k;
  i = j = k = 0;

  int first = 0;
  int post_type;

  // if no bias, first set to 1, load c from dst
  if (bias == nullptr) {
    first = 1;
  }

  for (k = 0; k < K; k += K_c) {
    if (k + K_c >= K) {
      post_type = act_type;
    } else {
      post_type = 0;
    }

    int cur_k = MIN(K - k, K_c);

    // pack b -> K_c * N;
    const float *pack_b_k = src_b + k * DivUp(N, n_block);

    for (i = 0; i < M; i += M_c) {
      int cur_m = MIN(M - i, M_c);
      // pack a -> M_c * K_c;
      pack_col_a_t(src_a + k + i * lda, lda, src_trans_buf, K_c, cur_k, cur_m,
                   conv_gemm_conf);

      for (j = 0; j < N;) {
        int cur_n = MIN(N - j, conv_gemm_conf.kernel_n_r_);
        float *cur_c = dst + i + j * ldc;

        const float *packed_cur_b =
            pack_b_k + DivDown(j, n_block) * K_c + j % n_block;
        const float *cur_bias = bias + j;
        conv_sgemm_block_n(cur_m, cur_n, cur_k, src_trans_buf, lda,
                           packed_cur_b, ldb, cur_c, ldc, cur_bias, first,
                           post_type, conv_gemm_conf);
        j += cur_n;
      }
    }
    // if k != 0, first = 1
    first = 1;
  }
}

// sgemm col_major a trans, b no_trans
// src_a: K * M, lda = K, prepacked
// src_b: K * N, ldb = K
// dst  : M * N, ldc = M
void conv_sgemm_tn_col_major_prepack_a(
    int M, int N, int K, const float *src_a, int lda, const float *src_b,
    int ldb, float *dst, int ldc, const float *bias, int act_type,
    float *pack_b_buf, conv_gemm_config<float, float, float> &conv_gemm_conf) {
  int M_c = conv_gemm_conf.M_c_;
  int K_c = conv_gemm_conf.K_c_;
  int m_block = conv_gemm_conf.m_block_;
  int n_block = conv_gemm_conf.n_block_;

  int first = 0;
  int post_type;

  // if no bias, first set to 1, load c from dst
  if (bias == nullptr) {
    first = 1;
  }

  for (int k = 0; k < K; k += K_c) {
    if (k + K_c >= K) {
      post_type = act_type;
    } else {
      post_type = 0;
    }

    int cur_k = MIN(K - k, K_c);

    // pack b -> K_c * N;
    pack_col_b_n(src_b + k, ldb, pack_b_buf, K_c, cur_k, N, conv_gemm_conf);

    OMP_PARALLEL_FOR_DYNAMIC_
    for (int i = 0; i < M; i += M_c) {
      int cur_m = MIN(M - i, M_c);
      // pack a -> M_c * K_c;
      auto src_a_i = src_a + k * DivUp(M, m_block) + i * K_c;

      for (int j = 0; j < N;) {
        int cur_n = MIN(N - j, conv_gemm_conf.kernel_n_r_);
        float *cur_c = dst + i + j * ldc;

        const float *packed_cur_b =
            pack_b_buf + DivDown(j, n_block) * K_c + j % n_block;
        const float *cur_bias = bias + j;
        conv_sgemm_block_n(cur_m, cur_n, cur_k, src_a_i, lda, packed_cur_b, ldb,
                           cur_c, ldc, cur_bias, first, post_type,
                           conv_gemm_conf);
        j += cur_n;
      }
    }
    // if k != 0, first = 1
    first = 1;
  }
}

// pack col major B no_trans [K x N]
void conv_pack_col_b_n(int N, int K, const float *src, int ld_src, float *dst,
                       conv_gemm_config<float, float, float> &conv_gemm_conf) {
  int K_c = conv_gemm_conf.K_c_;
  int n_block = conv_gemm_conf.n_block_;
  int N_round_up = DivUp(N, n_block);

  for (int k = 0; k < K; k += K_c) {
    int cur_k = MIN(K - k, K_c);
    float *pack_b = dst + k * N_round_up;
    pack_col_b_n(src + k, ld_src, pack_b, K_c, cur_k, N, conv_gemm_conf);
  }
}

// pack col major A trans [K x M]
void conv_pack_col_a_t(int M, int K, const float *src, int lda, float *dst,
                       conv_gemm_config<float, float, float> &conv_gemm_conf) {
  int M_c = conv_gemm_conf.M_c_;
  int K_c = conv_gemm_conf.K_c_;
  int m_block = conv_gemm_conf.m_block_;
  int n_block = conv_gemm_conf.n_block_;

  int i, j, k;
  i = j = k = 0;

  for (k = 0; k < K; k += K_c) {
    int cur_k = MIN(K - k, K_c);
    auto src_k = src + k;
    auto dst_k = dst + k * DivUp(M, m_block);

    for (i = 0; i < M; i += M_c) {
      int cur_m = MIN(M - i, M_c);
      // pack a -> M_c * K_c;
      pack_col_a_t(src_k + i * lda, lda, dst_k + i * K_c, K_c, cur_k, cur_m,
                   conv_gemm_conf);
    }
  }
}

// // pack A [K * M]
// void conv_pack_a_n()
// {

// }

void conv_ajust_m_blk_size(int max_num_threads, int m_all, int &m_blk) {
  // for 32bit, min M blk = 8
  // for 64bit, min M blk = 16
  int m_min = 8;
#ifdef XBYAK64
  m_min = 16;
#endif

  while ((m_all / m_blk) < max_num_threads && m_blk > m_min) {
    m_blk = MAX(m_blk / 2, m_min);
  }
}

}  // namespace jit
}  // namespace cpu
}  // namespace kernels
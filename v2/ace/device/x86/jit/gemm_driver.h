#pragma once

#include "config.h"

namespace kernels {
namespace cpu {
namespace jit {

// sgemm col_major a no_trans, b no_trans
void conv_sgemm_nn_col_major(
    int M, int N, int K, const float *src_a, int lda, const float *src_b,
    int ldb, float *dst, int ldc, const float *bias, int act_type,
    float *pack_buf, conv_gemm_config<float, float, float> &conv_gemm_conf);

// sgemm col_major a no_trans, b no_trans prepacked
void conv_sgemm_nn_col_major_prepack_b(
    int M, int N, int K, const float *src_a, int lda, const float *src_b,
    int ldb, float *dst, int ldc, const float *bias, int act_type,
    float *src_buf, conv_gemm_config<float, float, float> &conv_gemm_conf);

// sgemm col_major a trans, b no_trans prepacked
void conv_sgemm_tn_col_major_prepack_b(
    int M, int N, int K, const float *src_a, int lda, const float *src_b,
    int ldb, float *dst, int ldc, const float *bias, int act_type,
    float *src_trans_buf,
    conv_gemm_config<float, float, float> &conv_gemm_conf);

// sgemm col_major a trans prepacked, b no_trans
void conv_sgemm_tn_col_major_prepack_a(
    int M, int N, int K, const float *src_a, int lda, const float *src_b,
    int ldb, float *dst, int ldc, const float *bias, int act_type,
    float *src_trans_buf,
    conv_gemm_config<float, float, float> &conv_gemm_conf);

// sgemm col_major pack b no_trans
void conv_pack_col_b_n(int N, int K, const float *src, int ld_src, float *dst,
                       conv_gemm_config<float, float, float> &conv_gemm_conf);

// sgemm col_major pack a trans
void conv_pack_col_a_t(int M, int K, const float *src, int lda, float *dst,
                       conv_gemm_config<float, float, float> &conv_gemm_conf);

// adjust M block size (M_c_) for mutil-thread
void conv_ajust_m_blk_size(int max_num_threads, int m_all, int &m_blk);

using dim_t = long int;
void sgemm_nn_col_major(dim_t M, dim_t N, dim_t K, const float alpha,
                        const float *src_a, dim_t lda, const float *src_b,
                        dim_t ldb, const float beta, float *dst, dim_t ldc);
}  // namespace jit
}  // namespace cpu
}  // namespace kernels
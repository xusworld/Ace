#pragma once

#include "onednn.h"

namespace onednn {

// Create a _dynamic_ MatMul primitive that can work with arbitrary shapes
// and alpha parameters.
// Warning: current limitation is that beta parameter should be known in
// advance (use fixed_beta).
dnnl::matmul dynamic_matmul_create() {
  // We assume that beta is known at the primitive creation time
  float beta = 0.0f;

  dnnl::memory::dims a_shape = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
  dnnl::memory::dims b_shape = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
  dnnl::memory::dims c_shape = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};

  dnnl::memory::dims a_strides = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
  dnnl::memory::dims b_strides = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
  dnnl::memory::dims c_strides = {DNNL_RUNTIME_DIM_VAL, 1};

  dnnl::memory::desc a_md(a_shape, dnnl::memory::data_type::f32, a_strides);
  dnnl::memory::desc b_md(b_shape, dnnl::memory::data_type::f32, b_strides);
  dnnl::memory::desc c_md(c_shape, dnnl::memory::data_type::f32, c_strides);

  // Create attributes (to handle alpha dynamically and beta if necessary)
  dnnl::primitive_attr attr;
  attr.set_output_scales(/* mask */ 0, {DNNL_RUNTIME_F32_VAL});
  if (beta != 0.f) {
    post_ops po;
    po.append_sum(beta);
    attr.set_post_ops(po);
  }

  // Create a MatMul primitive
  dnnl::matmul::desc matmul_d(a_md, b_md, c_md);
  dnnl::matmul::primitive_desc matmul_pd(matmul_d, attr, eng);
  return matmul(matmul_pd);
}

// Create and execute a _static_ MatMul primitive. All shapes and parameters
// are hard-coded in the primitive and cannot be changed later.
void static_matmul_create_and_execute(char transA, char transB, int64_t M,
                                      int64_t N, int64_t K, float alpha,
                                      const float *A, int64_t lda,
                                      const float *B, int64_t ldb, float beta,
                                      float *C, int64_t ldc) {
  using dims = dnnl::memory::dims;

  // Prepare strides based on the transA and transB flags: transposed
  // matrices have strides swapped
  dims a_strides = tolower(transA) == 'n' ? dims{lda, 1} : dims{1, lda};
  dims b_strides = tolower(transB) == 'n' ? dims{ldb, 1} : dims{1, ldb};

  // Prepare memory descriptors
  dnnl::memory::desc a_md({M, K}, dnnl::memory::data_type::f32, a_strides);
  dnnl::memory::desc b_md({K, N}, dnnl::memory::data_type::f32, b_strides);
  dnnl::memory::desc c_md({M, N}, dnnl::memory::data_type::f32, {ldc, 1});

  // Create attributes (to handle alpha and beta if necessary)
  dnnl::primitive_attr attr;
  if (alpha != 1.f) attr.set_output_scales(/* mask */ 0, {alpha});
  if (beta != 0.f) {
    dnnl::post_ops po;
    po.append_sum(beta);
    attr.set_post_ops(po);
  }

  // Create a MatMul primitive
  dnnl::matmul::desc matmul_d(a_md, b_md, c_md);
  dnnl::matmul::primitive_desc matmul_pd(matmul_d, attr, eng);
  dnnl::matmul matmul_p(matmul_pd);

  // Wrap raw pointers into oneDNN memory objects
  dnnl::memory A_m(a_md, eng, (void *)A);
  dnnl::memory B_m(b_md, eng, (void *)B);
  dnnl::memory C_m(c_md, eng, (void *)C);

  // Execute the MatMul primitive.
  // Since here all shapes and parameters are static, please note that we
  // don't need to pass alpha (scales) again, as they are already hard-coded
  // in the primitive descriptor. Also, we are not allowed to change the
  // shapes of matrices A, B, and C -- they should exactly match
  // the memory descriptors passed to MatMul operation descriptor.
  dnnl::stream s(eng);
  matmul_p.execute(
      s, {{DNNL_ARG_SRC, A_m}, {DNNL_ARG_WEIGHTS, B_m}, {DNNL_ARG_DST, C_m}});
  s.wait();
}

}  // namespace onednn
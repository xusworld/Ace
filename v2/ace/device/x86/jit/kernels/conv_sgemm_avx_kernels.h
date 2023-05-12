#pragma once

#include <immintrin.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <xbyak/xbyak.h>
#include <xmmintrin.h>

#include <exception>
#include <fstream>
#include <utility>

#include "conv_sgemm_avx_16_i.h"
#include "conv_sgemm_avx_1_i.h"
#include "conv_sgemm_avx_2_i.h"
#include "conv_sgemm_avx_4_i.h"
#include "conv_sgemm_avx_8_i.h"
#include "cpu/x64/jit/common/abi_info.h"
#include "cpu/x64/jit/common/asm_common.h"
#include "cpu/x64/jit/jit_code_generator.h"

namespace kernels {
namespace cpu {
namespace jit {

template <int M = 8, int N = 6, int M_BLOCK_SIZE = 16, int N_BLOCK_SIZE = 6>
class conv_sgemm_avx_kernel : public JitCodeGenerator {
 public:
  static void naive_impl(const int K, const float* src_a, const int lda,
                         const float* src_b, int ldb, float* dst, int ldc,
                         const float* bias, int first, int act_type) {}

  using func_ptr_t = decltype(&conv_sgemm_avx_kernel::naive_impl);

 public:
  conv_sgemm_avx_kernel() {
    switch (M) {
      case 1:
        actual = new conv_sgemm_avx_1xi<N, M_BLOCK_SIZE, N_BLOCK_SIZE>();
        break;
      case 2:
        actual = new conv_sgemm_avx_2xi<N, M_BLOCK_SIZE, N_BLOCK_SIZE>();
        break;
      case 4:
        actual = new conv_sgemm_avx_4xi<N, M_BLOCK_SIZE, N_BLOCK_SIZE>();
        break;
      case 8:
        actual = new conv_sgemm_avx_8xi<N, M_BLOCK_SIZE, N_BLOCK_SIZE>();
        break;
      case 16:
        actual = new conv_sgemm_avx_16xi<N, M_BLOCK_SIZE, N_BLOCK_SIZE>();
        break;
      default:
        throw std::runtime_error("kernel not found for specified param.");
        break;
    }
    ret();
  }

  virtual void* get_func_ptr() {
    if (actual != nullptr) {
      return actual->get_func_ptr();
    } else {
      throw std::runtime_error("kernel not initialized.");
    }
    return JitCodeGenerator::get_func_ptr();
  }

  virtual std::string get_kernel_name() {
    if (actual) {
      return actual->get_kernel_name();
    } else {
      throw std::runtime_error("kernel not initialized.");
    }
    return JIT_KERNEL_NAME(conv_sgemm_avx_kernel);
  }

  virtual size_t get_func_size() {
    if (actual) {
      return actual->get_func_size();
    } else {
      throw std::runtime_error("kernel not initialized.");
    }
    return JitCodeGenerator::get_func_size();
  }

  virtual ~conv_sgemm_avx_kernel() {
    if (actual != nullptr) {
      delete actual;
      actual = nullptr;
    }
  }

 private:
  JitCodeGenerator* actual = nullptr;
};

}  // namespace jit
}  // namespace cpu
}  // namespace kernels
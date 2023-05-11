#include "config.h"

#include <memory>
#include <mutex>

#include "cpu/x64/jit/kernels/conv_sgemm_avx_kernels.h"
#include "cpu/x64/jit/kernels/sgemm_fetch_n.h"
#include "cpu/x64/jit/kernels/sgemm_fetch_n_6.h"
#include "cpu/x64/jit/kernels/sgemm_fetch_t.h"
#include "cpu/x64/jit/kernels/sgemm_fetch_t_16.h"
#include "cpu/x64/jit/kernels/sgemm_fetch_t_4.h"
#include "cpu/x64/jit/kernels/sgemm_fetch_t_4x16.h"
#include "cpu/x64/jit/kernels/sgemm_fetch_t_8.h"

namespace kernels {
namespace cpu {
namespace jit {

template <typename a_t, typename b_t, typename c_t>
conv_gemm_config<a_t, b_t, c_t>::conv_gemm_config(const int m_block,
                                                  const int n_block)
    : m_block_(m_block), n_block_(n_block) {
  std::vector<int> supported_m_block = {4, 8, 16};
  std::vector<int> supported_n_block = {6};
  if (std::find(supported_m_block.begin(), supported_m_block.end(), m_block_) ==
      supported_m_block.end()) {
    throw std::runtime_error("value of m_block is not supported.");
  }
  if (std::find(supported_n_block.begin(), supported_n_block.end(), n_block_) ==
      supported_n_block.end()) {
    throw std::runtime_error("value of n_block is not supported.");
  }

  M_c_ = 64;
  K_c_ = 256;

  if (cpu_with_isa(avx2)) {
#ifdef XBYAK64
    m_block_ = 16;
    kernel_m_r_ = 16;
#else
    m_block_ = 8;
    kernel_m_r_ = 8;
#endif
  } else if (cpu_with_isa(sse42)) {
    m_block_ = 4;
    kernel_m_r_ = 4;
  }
  kernel_n_r_ = 6;
  this->init_jit_kernel();
}

template conv_gemm_config<float, float, float>::conv_gemm_config(
    const int m_block, const int n_block);

template <typename a_t, typename b_t, typename c_t>
void conv_gemm_config<a_t, b_t, c_t>::init_jit_kernel() {
  static std::shared_ptr<JitCodeGenerator> g_pack_t_ker[nb_kernels_m + 1];
  static std::shared_ptr<JitCodeGenerator> g_pack_t_4x16_ker;
  static std::shared_ptr<JitCodeGenerator> g_pack_n_ker[nb_kernels_n + 1];
  static std::shared_ptr<JitCodeGenerator> g_kernel_4[nb_kernels_m + 1]
                                                     [nb_kernels_n + 1];
  static std::shared_ptr<JitCodeGenerator> g_kernel_8[nb_kernels_m + 1]
                                                     [nb_kernels_n + 1];
  static std::shared_ptr<JitCodeGenerator> g_kernel_16[nb_kernels_m + 1]
                                                      [nb_kernels_n + 1];

  static std::once_flag initialized;
  std::call_once(initialized, [] {
    // initialize the pack t kernels
    for (int i = 1; i <= nb_kernels_n; i++) {
      switch (i) {
        case 1:
          g_pack_n_ker[i] = std::make_shared<sgemm_fetch_n_1_ker_t>();
          break;
        case 2:
          g_pack_n_ker[i] = std::make_shared<sgemm_fetch_n_2_ker_t>();
          break;
        case 3:
          g_pack_n_ker[i] = std::make_shared<sgemm_fetch_n_3_ker_t>();
          break;
        case 4:
          g_pack_n_ker[i] = std::make_shared<sgemm_fetch_n_4_ker_t>();
          break;
        case 5:
          g_pack_n_ker[i] = std::make_shared<sgemm_fetch_n_5_ker_t>();
          break;
        case 6:
          if (cpu_with_isa(avx2)) {
            g_pack_n_ker[i] = std::make_shared<sgemm_fetch_n_6_ker_t>();
          } else if (cpu_with_isa(sse42)) {
            g_pack_n_ker[i] = std::make_shared<sgemm_fetch_n_6i_ker_t>();
          }
          break;
        default:
          throw std::runtime_error("uninitialized kernel.");
          break;
      }
    }

    // initialize the pack t kernels
    for (int i = 1; i <= nb_kernels_m; i++) {
      switch (i) {
        case 0:
          break;
        case 1:
          g_pack_t_ker[i] = std::make_shared<sgemm_fetch_t_1_ker_t>();
          break;
        case 2:
          g_pack_t_ker[i] = std::make_shared<sgemm_fetch_t_2_ker_t>();
          break;
        case 3:
          g_pack_t_ker[i] = std::make_shared<sgemm_fetch_t_3_ker_t>();
          break;
        case 4:
          g_pack_t_ker[i] = std::make_shared<sgemm_fetch_t_4_ker_t>();
          break;
        case 8:
          g_pack_t_ker[i] = std::make_shared<sgemm_fetch_t_8_ker_t>();
          break;
        case 16:
          g_pack_t_ker[i] = std::make_shared<sgemm_fetch_t_16_ker_t>();
          break;
        default:
          break;
      }
    }

    g_pack_t_4x16_ker = std::make_shared<sgemm_fetch_t_4x16_ker_t>();

#define REGISTER_KERNEL(M, N)                                                 \
  if (M <= 4)                                                                 \
    g_kernel_4[M][N] = std::make_shared<conv_sgemm_avx_kernel<M, N, 4, 6>>(); \
  if (M <= 8)                                                                 \
    g_kernel_8[M][N] = std::make_shared<conv_sgemm_avx_kernel<M, N, 8, 6>>(); \
  if (M <= 16)                                                                \
    g_kernel_16[M][N] = std::make_shared<conv_sgemm_avx_kernel<M, N, 16, 6>>();

#define REGISTER_KERNEL_M(M) \
  REGISTER_KERNEL(M, 1);     \
  REGISTER_KERNEL(M, 2);     \
  REGISTER_KERNEL(M, 3);     \
  REGISTER_KERNEL(M, 4);     \
  REGISTER_KERNEL(M, 5);     \
  REGISTER_KERNEL(M, 6)

    REGISTER_KERNEL_M(1);
    REGISTER_KERNEL_M(2);
    REGISTER_KERNEL_M(4);
    REGISTER_KERNEL_M(8);
    REGISTER_KERNEL_M(16);

#ifdef TNN_JIT_DUMP_KERNEL
    for (int i = 1; i <= nb_kernels_m; i++) {
      if (g_pack_t_ker[i]) {
        g_pack_t_ker[i]->dump_to_file();
      }
    }
    g_pack_t_4x16_ker->dump_to_file();

    for (int i = 1; i <= nb_kernels_n; i++) {
      if (g_pack_n_ker[i]) {
        g_pack_n_ker[i]->dump_to_file();
      }
    }

    for (int m = 1; m <= nb_kernels_m; m++) {
      for (int n = 1; n <= nb_kernels_n; n++) {
        if (g_kernel_4[m][n]) {
          g_kernel_4[m][n]->dump_to_file();
        }
        if (g_kernel_8[m][n]) {
          g_kernel_8[m][n]->dump_to_file();
        }
        if (g_kernel_16[m][n]) {
          g_kernel_16[m][n]->dump_to_file();
        }
      }
    }
#endif
  });  // end of initialized

  for (int i = 1; i <= nb_kernels_m; i++) {
    if (g_pack_t_ker[i]) {
      pack_t_ker_[i] =
          get_func_ptr<sgemm_fetch_t_1_ker_t>(g_pack_t_ker[i].get());
    } else {
      pack_t_ker_[i] = nullptr;
    }
  }

  if (g_pack_t_4x16_ker) {
    pack_t_4x16_ker_ =
        get_func_ptr<sgemm_fetch_t_1_ker_t>(g_pack_t_4x16_ker.get());
  }

  for (int i = 1; i <= nb_kernels_n; i++) {
    if (g_pack_n_ker[i]) {
      pack_n_ker_[i] =
          get_func_ptr<sgemm_fetch_n_1_ker_t>(g_pack_n_ker[i].get());
    } else {
      pack_n_ker_[i] = nullptr;
    }
  }

  using kernel_array_ptr = decltype(&g_kernel_16);
  kernel_array_ptr kernel_array;
  if (m_block_ == 4) {
    kernel_array = &g_kernel_4;
  } else if (m_block_ == 8) {
    kernel_array = &g_kernel_8;
  } else if (m_block_ == 16) {
    kernel_array = &g_kernel_16;
  } else {
    throw std::runtime_error("unsupported m_block value.");
  }

  for (int m = 1; m <= nb_kernels_m; m++) {
    for (int n = 1; n <= nb_kernels_n; n++) {
      if ((*kernel_array)[m][n]) {
        kernels_[m][n] = get_func_ptr<conv_sgemm_avx_kernel<0, 0>>(
            (*kernel_array)[m][n].get());
      } else {
        kernels_[m][n] = nullptr;
      }
    }
  }
}

template void conv_gemm_config<float, float, float>::init_jit_kernel();

}  // namespace jit
}  // namespace cpu
}  // namespace kernels
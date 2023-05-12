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

#include "cpu/x64/jit/common/abi_info.h"
#include "cpu/x64/jit/common/asm_common.h"
#include "cpu/x64/jit/jit_code_generator.h"

namespace kernels {
namespace cpu {
namespace jit {

template <int I, int M_BLOCK_SIZE, int N_BLOCK_SIZE>
class conv_sgemm_avx_16xi : public JitCodeGenerator {
 public:
  static void naive_impl(const int K, const float *src_a, const int lda,
                         const float *src_b, int ldb, float *dst, int ldc,
                         const float *bias, int first, int act_type) {}

  using func_ptr_t = decltype(&conv_sgemm_avx_16xi::naive_impl);

  virtual std::string get_kernel_name() {
    std::stringstream buf;
    buf << JIT_KERNEL_NAME(conv_sgemm_avx_16) << "_" << I << "_" << M_BLOCK_SIZE
        << "_" << N_BLOCK_SIZE;
    return buf.str();
  }

 public:
  conv_sgemm_avx_16xi() {
#ifdef XBYAK64
    constexpr int N_r = MIN_(6, I);

    declare_param<const long int>();  // 0. K
    declare_param<const float *>();   // 1. src_a
    declare_param<const long int>();  // 2. lda
    declare_param<const float *>();   // 3. src_b
    declare_param<const long int>();  // 4. ldb
    declare_param<float *>();         // 5. dst
    declare_param<const long int>();  // 6. ldc
    declare_param<const float *>();   // 7. bias
    declare_param<long int>();        // 8. first
    declare_param<long int>();        // 9. act_type

    abi_prolog();

    stack_var K = get_arguement_to_stack(0);
    reg_var src_a = get_arguement(1);
    reg_var lda = get_arguement(2);
    reg_var src_b = get_arguement(3);
    reg_var ldb = get_arguement(4);
    reg_var dst = get_arguement(5);
    reg_var ldc = get_arguement(6);
    reg_var bias = get_arguement(7);
    reg_var first = get_arguement(8);
    reg_var act_type = get_arguement(9);

    reg_var c[3] = {REG_VAR_ARRAY_3};
    reg_var op_6f(this);
    vreg_var v_const(this);
    vreg_var c_data[2][6] = {{VREG_VAR_ARRAY_6}, {VREG_VAR_ARRAY_6}};
    vreg_var a_data[2] = {VREG_VAR_ARRAY_2};
    vreg_var b_data[2] = {VREG_VAR_ARRAY_2};

    ldc.restore();
    mov(c[0].aquire(), dst.restore());
    lea(c[1].aquire(), byte[dst + (ldc * 8)]);
    lea(c[2].aquire(), byte[c[1] + (ldc * 8)]);
    dst.release();

    Xbyak::RegExp c_addr[6] = {
        Xbyak::RegExp(c[0]), Xbyak::RegExp(c[0] + (ldc * 4)),
        Xbyak::RegExp(c[1]), Xbyak::RegExp(c[1] + (ldc * 4)),
        Xbyak::RegExp(c[2]), Xbyak::RegExp(c[2] + (ldc * 4)),
    };

    first.restore();
    cmp(first, 0);
    jne("L_init");
    bias.restore();
    for (int i = 0; i < N_r; i++) {
      vbroadcastss(c_data[0][i].aquire(), dword[bias + i * 4]);
      vbroadcastss(c_data[1][i].aquire(), dword[bias + i * 4]);
    }
    bias.release();
    jmp("L_init_end");
    L("L_init");
    for (int i = 0; i < N_r; i++) {
      vmovups(c_data[0][i], yword[c_addr[i]]);
      vmovups(c_data[1][i], yword[c_addr[i] + 8 * 4]);
    }
    L("L_init_end");
    first.release();

    src_a.restore();
    src_b.restore();

    LOOP_STACK_VAR(K, SGEMM_AVX_8X6_K) {
      vmovaps(a_data[0].aquire(), yword[src_a]);
      vmovaps(a_data[1].aquire(), yword[src_a + 8 * 4]);

      for (int i = 0; i < N_r; i += 2) {
        vbroadcastss(b_data[0].aquire(), yword[src_b + i * 4]);
        vfmadd231ps(c_data[0][i], a_data[0], b_data[0]);
        vfmadd231ps(c_data[1][i], a_data[1], b_data[0].release());

        if (i + 1 < N_r) {
          vbroadcastss(b_data[1].aquire(), yword[src_b + i * 4 + 4]);
          vfmadd231ps(c_data[0][i + 1], a_data[0], b_data[1]);
          vfmadd231ps(c_data[1][i + 1], a_data[1], b_data[1].release());
        }
      }

      a_data[0].release();
      a_data[1].release();

      lea(src_a, byte[src_a + M_BLOCK_SIZE * 4]);
      lea(src_b, byte[src_b + N_BLOCK_SIZE * 4]);
    }

    src_a.release();
    src_b.release();

    // only support fuse relu, relu6
    act_type.restore();
    cmp(act_type, 0);
    je("L_post_end_1");
    v_const.aquire();
    vxorps(v_const, v_const, v_const);
    for (int i = 0; i < N_r; i++) {
      vmaxps(c_data[0][i], c_data[0][i], v_const);
      vmaxps(c_data[1][i], c_data[1][i], v_const);
    }
    v_const.release();
    L("L_post_end_1");

    cmp(act_type, 2);
    jne("L_post_end_2");
    op_6f.restore();
    v_const.aquire();
    // 6.f
    mov(op_6f.cvt32(), 0x40C00000);
    movd(v_const.xmm(), op_6f.cvt32());
    vbroadcastss(v_const, v_const.xmm());
    for (int i = 0; i < N_r; i++) {
      vminps(c_data[0][i], c_data[0][i], v_const);
      vminps(c_data[1][i], c_data[1][i], v_const);
    }
    v_const.release();
    op_6f.release();
    L("L_post_end_2");
    act_type.release();

    for (int i = 0; i < N_r; i++) {
      vmovups(yword[c_addr[i]], c_data[0][i]);
      vmovups(yword[c_addr[i] + 8 * 4], c_data[1][i]);
    }

    abi_epilog();
#endif  // XBYAK64
    ret();
  }

  virtual ~conv_sgemm_avx_16xi() {}

 private:
};

}  // namespace jit
}  // namespace cpu
}  // namespace kernels
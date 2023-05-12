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

#include "cpu/x64/jit/common/asm_common.h"
#include "cpu/x64/jit/jit_code_generator.h"

namespace kernels {
namespace cpu {
namespace jit {
//  only support block_size = 6
class sgemm_fetch_n_6_ker_t : public JitCodeGenerator {
 public:
  static void naive_impl(const int m, const float *a, const int lda, float *b,
                         const int ldb, const int block_size) {}

  using func_ptr_t = decltype(&sgemm_fetch_n_6_ker_t::naive_impl);

  virtual std::string get_kernel_name() {
    return JIT_KERNEL_NAME(sgemm_fetch_n_6);
  }

 public:
  sgemm_fetch_n_6_ker_t() {
    // 校验参数
    declare_param<const long int>();
    declare_param<const float *>();
    declare_param<const long int>();
    declare_param<float *>();
    declare_param<const long int>();
    declare_param<const long int>();

    // ABI
    abi_prolog();

    // 参数
    stack_var m = get_arguement_to_stack(0);
    stack_var a_ptr = get_arguement_to_stack(1);
    stack_var lda = get_arguement_to_stack(2);
    stack_var b_ptr = get_arguement_to_stack(3);
    stack_var ldb = get_arguement_to_stack(4);
    reg_var block_size = get_arguement(5);

    stack_var m8 = get_stack_var();
    stack_var m1 = get_stack_var();

    reg_var tmp(this), a0(this), a1(this), a2(this), a3(this), b_ptr_reg(this);
    stack_var a4 = get_stack_var();
    stack_var a5 = get_stack_var();

    reg_var ldax4(this);

    // init m8 = m / 8
    mov(tmp.aquire(), m);
    sar(tmp, 0x3);
    mov(m8, tmp);

    // init m1 = m % 8
    mov(tmp, m);
    and_(tmp, 0x7);
    mov(m1, tmp);

    block_size.restore();
    cmp(block_size, 0x6);
    block_size.release();
    jne("FUNCTION_END", T_NEAR);

    // init ldax4 = lda * sizeof(float);
    mov(ldax4.aquire(), lda);
    lea(ldax4, qword[ldax4 * sizeof(float)]);

    // init a pointers
    mov(tmp, a_ptr);
    mov(a0.aquire(), tmp);

    add(tmp, ldax4);
    mov(a1.aquire(), tmp);

    add(tmp, ldax4);
    mov(a2.aquire(), tmp);

    add(tmp, ldax4);
    mov(a3.aquire(), tmp);

    add(tmp, ldax4);
    mov(a4, tmp);

    add(tmp, ldax4.release());
    mov(a5, tmp);

    vreg_var v0(this), v1(this), v2(this), v3(this), v4(this), v5(this);
    vreg_var upack0(this), upack1(this), upack2(this), upack3(this),
        upack4(this), upack5(this);
    vreg_var shf0(this), shf1(this), shf2(this), shf3(this), shf4(this),
        shf5(this);
    vreg_var res0(this), res1(this), res2(this), res3(this), res4(this),
        res5(this);

    // 循环 m / 8 次
    LOOP_STACK_VAR(m8, SGEMM_FETCH_N6_M8) {
      // 记录指针位置
      vmovups(v0.aquire(), yword[a0]);
      vmovups(v1.aquire(), yword[a1]);
      vmovups(v2.aquire(), yword[a2]);
      vmovups(v3.aquire(), yword[a3]);
      mov(tmp, a4);
      vmovups(v4.aquire(), yword[tmp]);
      mov(tmp, a5);
      vmovups(v5.aquire(), yword[tmp]);

      vunpcklps(upack0.aquire(), v0, v1);
      vunpckhps(upack1.aquire(), v0.release(), v1.release());
      vunpcklps(upack2.aquire(), v2, v3);
      vunpckhps(upack3.aquire(), v2.release(), v3.release());
      vunpcklps(upack4.aquire(), v4, v5);
      vunpckhps(upack5.aquire(), v4.release(), v5.release());

      // 0x44 = 0b 01 00 01 00 --> _MM_SHUFFLE(1, 0, 1, 0)
      vshufps(shf0.aquire(), upack0, upack2, 0x44);
      vshufps(shf1.aquire(), upack4, upack0.release(), 0xe4);
      vshufps(shf2.aquire(), upack2.release(), upack4.release(), 0xee);
      vshufps(shf3.aquire(), upack5, upack1, 0xe4);
      vshufps(shf4.aquire(), upack3, upack5.release(), 0xee);
      vshufps(shf5.aquire(), upack1.release(), upack3.release(), 0x44);

      vinsertf128(res0.aquire(), shf0, Xbyak::Xmm(shf1.getIdx()), 0x1);
      vperm2f128(res1.aquire(), shf0.release(), shf1.release(), 0x31);
      vinsertf128(res2.aquire(), shf2, Xbyak::Xmm(shf5.getIdx()), 0x1);
      vperm2f128(res3.aquire(), shf2.release(), shf5.release(), 0x31);
      vinsertf128(res4.aquire(), shf3, Xbyak::Xmm(shf4.getIdx()), 0x1);
      vperm2f128(res5.aquire(), shf3.release(), shf4.release(), 0x31);

      mov(tmp, b_ptr);
      size_t vsize_in_bytes = 8 * sizeof(float);  // 32 bytes
      vmovups(yword[tmp + 0 * vsize_in_bytes], res0);
      vmovups(yword[tmp + 1 * vsize_in_bytes], res2);
      vmovups(yword[tmp + 2 * vsize_in_bytes], res4);
      vmovups(yword[tmp + 3 * vsize_in_bytes], res1);
      vmovups(yword[tmp + 4 * vsize_in_bytes], res3);
      vmovups(yword[tmp + 5 * vsize_in_bytes], res5);

      add(b_ptr, 6 * vsize_in_bytes);

      // jump to another 8 float point values
      add(a0, vsize_in_bytes);
      add(a1, vsize_in_bytes);
      add(a2, vsize_in_bytes);
      add(a3, vsize_in_bytes);
      add(a4, vsize_in_bytes);
      add(a5, vsize_in_bytes);
    }

    size_t ele_size = sizeof(float);
    mov(b_ptr_reg.aquire(), b_ptr);
    LOOP_STACK_VAR(m1, SGEMM_FETCH_N6_M1) {
      mov(tmp.cvt32(), dword[a0]);
      mov(dword[b_ptr_reg + 0 * ele_size], tmp.cvt32());

      mov(tmp.cvt32(), dword[a1]);
      mov(dword[b_ptr_reg + 1 * ele_size], tmp.cvt32());

      mov(tmp.cvt32(), dword[a2]);
      mov(dword[b_ptr_reg + 2 * ele_size], tmp.cvt32());

      mov(tmp.cvt32(), dword[a3]);
      mov(dword[b_ptr_reg + 3 * ele_size], tmp.cvt32());

      mov(tmp, a4);
      mov(tmp.cvt32(), dword[tmp]);
      mov(dword[b_ptr_reg + 4 * ele_size], tmp.cvt32());

      mov(tmp, a5);
      mov(tmp.cvt32(), dword[tmp]);
      mov(dword[b_ptr_reg + 5 * ele_size], tmp.cvt32());

      add(a0, sizeof(float));
      add(a1, sizeof(float));
      add(a2, sizeof(float));
      add(a3, sizeof(float));
      add(a4, sizeof(float));
      add(a5, sizeof(float));
      add(b_ptr_reg, 6 * sizeof(float));
    }

    L("FUNCTION_END");
    abi_epilog();
    ret();
  }

  virtual ~sgemm_fetch_n_6_ker_t() {}

 private:
};

}  // namespace jit
}  // namespace cpu
}  // namespace kernels

/*
            a11, a12, a13, a14, a15, a16, a17, a18
            a21, a22, a23, a24, a25, a26, a27, a28
            a31, a32, a33, a34, a35, a36, a37, a38
            a41, a42, a43, a44, a45, a46, a47, a48
            a51, a52, a53, a54, a55, a56, a57, a58
            a61, a62, a63, a64, a65, a66, a67, a68

            a18, a17, a16, a15, a14, a13, a12, a11   v0
            a28, a27, a26, a25, a24, a23, a22, a21   v1
            a38, a37, a36, a35, a34, a33, a32, a31   v2
            a48, a47, a46, a45, a44, a43, a42, a41   v3
            a58, a57, a56, a55, a54, a53, a52, a51   v4
            a68, a67, a66, a65, a64, a63, a62, a61   v5

            a24, a14, a23, a13, a22, a12, a21, a11   vunpcklps unpack0, v0, v1
            a28, a18, a27, a17, a26, a16, a25, a15   vunpckhps unpack1, v0, v1
            a44, a34, a43, a33, a42, a32, a41, a31   vunpcklps unpack2, v2, v3
            a48, a38, a47, a37, a46, a36, a45, a35   vunpckhps unpack3, v2, v3
            a64, a54, a63, a53, a62, a52, a61, a51   vunpcklps unpack4, v4, v5
            a68, a58, a67, a57, a66, a56, a65, a55   vunpckhps unpack5, v4, v5

            a33, a43, a13, a23, a31, a41, a11, a21    shf0: vshufps upack0,
   upack2, 0b01000100 (1, 0, 1, 0) a13, a23, a53, a63, a12, a22, a52, a62 shf1:
   vshufps upack4, upack0, 0b11100100 (3, 2, 1, 0) shf2: vshufps upack2, upack4,
   0b11101110 (3, 2, 3, 2) shf3: vshufps upack5, upack1, 0b11100100 (3, 2, 1, 0)
                                                     shf4: vshufps upack3,
   upack5, 0b11101110 (3, 2, 3, 2) shf5: vshufps upack1, upack3, 0b01000100 (1,
   0, 1, 0)

      */
// vshufps(shf0.aquire(), upack0, upack2, 0x44);
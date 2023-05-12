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

template <int I>
class sgemm_fetch_t_i_ker_t : public JitCodeGenerator {
 public:
  static void naive_impl(const int m, const float *a, const int lda, float *b,
                         const int ldb, const int block_size) {}

  using func_ptr_t = decltype(&sgemm_fetch_t_i_ker_t::naive_impl);

  virtual std::string get_kernel_name() {
    std::stringstream buf;
    buf << JIT_KERNEL_NAME(sgemm_fetch_t_) << I;
    return buf.str();
  }

 public:
  sgemm_fetch_t_i_ker_t() {
    declare_param<const size_t>();
    declare_param<const float *>();
    declare_param<const size_t>();
    declare_param<float *>();
    declare_param<const long int>();
    declare_param<const long int>();

    abi_prolog();

    stack_var m = get_arguement_to_stack(0);
    stack_var a_stack = get_arguement_to_stack(1);
    stack_var lda = get_arguement_to_stack(2);
    stack_var b_stack = get_arguement_to_stack(3);
    stack_var ldb = get_arguement_to_stack(4);
    stack_var block_size = get_arguement_to_stack(5);

    stack_var m4 = get_stack_var();
    stack_var m1 = get_stack_var();

    reg_var tmp(this);
    reg_var a_r[2] = {REG_VAR_ARRAY_2};
    reg_var b_r(this);

    reg_var ldax4(this);
    reg_var ldax8(this);
    reg_var block_size_x4(this);

    // init m8 = m / 4
    mov(tmp.aquire(), m);
    sar(tmp, 0x2);
    mov(m4, tmp);

    // init m1 = m % 4
    mov(tmp, m);
    and_(tmp, 0x3);
    mov(m1, tmp);

    mov(b_r.aquire(), b_stack);
    b_r.stash();

    mov(ldax4.aquire(), lda);
    lea(ldax4, qword[ldax4 * sizeof(float)]);
    ldax4.store();

    // init a pointers
    mov(tmp, a_stack);
    mov(a_r[0].aquire(), tmp);
    add(tmp, ldax4);
    mov(a_r[1].aquire(), tmp);
    tmp.release();

    // lda for 4 lines
    lea(ldax8.aquire(), qword[ldax4.release() * 2]);
    ldax8.stash();

    mov(block_size_x4.aquire(), block_size);
    lea(block_size_x4, qword[block_size_x4 * sizeof(float)]);
    block_size_x4.stash();

    stack_var a_data[4][8] = {
        {STACK_VAR_ARRAY_8},
        {STACK_VAR_ARRAY_8},
        {STACK_VAR_ARRAY_8},
        {STACK_VAR_ARRAY_8},
    };

    LOOP_STACK_VAR(m4, SGEMM_FETCH_TI_M4) {
      // load
      ldax8.restore();
      Xbyak::RegExp a_addr[4] = {
          Xbyak::RegExp(a_r[0]),
          Xbyak::RegExp(a_r[1]),
          Xbyak::RegExp(a_r[0] + ldax8),
          Xbyak::RegExp(a_r[1] + ldax8),
      };

      for (int line = 0; line < 4; line++) {
        for (int i = 0; i < I; i++) {
          mov(tmp.aquire().cvt32(), dword[a_addr[line] + i * sizeof(float)]);
          mov(a_data[line][i], tmp.release().cvt32());
        }
      }
      lea(a_r[0], byte[a_r[0] + ldax8 * 2]);
      lea(a_r[1], byte[a_r[1] + ldax8 * 2]);

      ldax8.release();

      // store
      b_r.restore();
      for (int line = 0; line < 4; line++) {
        for (int i = 0; i < I; i++) {
          mov(tmp.aquire().cvt32(), a_data[line][i]);
          mov(dword[b_r + i * sizeof(float)], tmp.release().cvt32());
        }
        lea(b_r, byte[b_r + block_size_x4.restore()]);
        block_size_x4.release();
      }
      b_r.stash();
    }

    LOOP_STACK_VAR(m1, SGEMM_FETCH_TI_M1) {
      for (int i = 0; i < I; i++) {
        mov(tmp.aquire().cvt32(), dword[a_r[0] + i * sizeof(float)]);
        mov(a_data[0][i], tmp.release().cvt32());
      }
      lea(a_r[0], byte[a_r[0] + ldax4.restore()]);
      ldax4.release();

      b_r.restore();
      for (int i = 0; i < I; i++) {
        mov(tmp.aquire().cvt32(), a_data[0][i]);
        mov(dword[b_r + i * sizeof(float)], tmp.release().cvt32());
      }
      lea(b_r, byte[b_r + block_size_x4.restore()]);
      b_r.stash();
    }

    abi_epilog();
    ret();
  }

  virtual ~sgemm_fetch_t_i_ker_t() {}

 private:
};

class sgemm_fetch_t_1_ker_t : public sgemm_fetch_t_i_ker_t<1> {};
class sgemm_fetch_t_2_ker_t : public sgemm_fetch_t_i_ker_t<2> {};
class sgemm_fetch_t_3_ker_t : public sgemm_fetch_t_i_ker_t<3> {};
// class sgemm_fetch_t_4_ker_t : public sgemm_fetch_t_i_ker_t<4> {};
class sgemm_fetch_t_5_ker_t : public sgemm_fetch_t_i_ker_t<5> {};
class sgemm_fetch_t_6_ker_t : public sgemm_fetch_t_i_ker_t<6> {};
class sgemm_fetch_t_7_ker_t : public sgemm_fetch_t_i_ker_t<7> {};

}  // namespace jit
}  // namespace cpu
}  // namespace kernels
#pragma once

#include <immintrin.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <xbyak/xbyak.h>
#include <xmmintrin.h>

#include <algorithm>
#include <exception>
#include <fstream>
#include <queue>
#include <sstream>
#include <stack>
#include <string>

#include "cpu/x64/jit/common/abi_info.h"
#include "cpu/x64/jit/common/asm_common.h"
#include "cpu/x64/jit/cpu_isa.h"
#include "macros.h"

namespace kernels {
namespace cpu {
namespace jit {

class JitCodeGenerator;

template <typename T>
typename T::func_ptr_t get_func_ptr(T* jit_kernel) {
  typename T::func_ptr_t ptr;
  T* our_kernel = dynamic_cast<T*>(jit_kernel);
  if (!our_kernel) {
    throw std::runtime_error("Not a type of JitCodeGenerator");
  }
  const void* jit_func_addr = our_kernel->get_func_ptr();
  if (jit_func_addr == nullptr) {
    throw std::runtime_error("getCode() got null ptr");
  }
  memcpy(&ptr, &jit_func_addr, sizeof(void*));
  return ptr;
}

template <typename T>
typename T::func_ptr_t get_func_ptr(JitCodeGenerator* jit_kernel) {
  return get_func_ptr<T>((T*)jit_kernel);
}

class JitCodeGenerator : public Xbyak::CodeGenerator {
 public:
  enum WHEN_TO_USE_FLAG : int32_t {
    NOW = 0,
    LATER = 1,
  };

  struct stack_var : public Xbyak::Address {
    explicit stack_var(Xbyak::Reg64 base, size_t offset,
                       uint32_t bit = rword_.bit_)
        : Xbyak::Address(bit, false, base + offset) {}
  };

  struct reg_var : public Xbyak::Reg64 {
    explicit reg_var(JitCodeGenerator* ker)
        : ker_(ker), v_stack_(ker->get_stack_var()) {}
    explicit reg_var(JitCodeGenerator* ker, stack_var var_s)
        : ker_(ker), v_stack_(var_s) {}

    ~reg_var() {
      if (in_use_) this->release();
    }

    // get a real register
    Xbyak::Reg64 aquire() {
      if (!in_use_) {
        Xbyak::Reg64 rf = this->ker_->get_free_reg();
        setIdx(rf.getIdx());
        setKind(rf.getKind());
        setBit(rf.getBit());
        setOpmaskIdx(rf.getOpmaskIdx());
        setRounding(rf.getRounding());
        zero_ = rf.hasZero();
        in_use_ = true;
      } else {
        throw std::runtime_error("the register is already in use!");
      }
      return *this;
    }

    // release the real register
    Xbyak::Reg64 release() {
      if (in_use_) {
        in_use_ = false;
        this->ker_->drop_reg(*this);
      } else {
        throw std::runtime_error(
            "the register is not in use, can't be relased.");
      }
      return *this;
    }

    // save the content to stack, then release the real register
    void stash() {
      if (in_use_) {
        ker_->mov(v_stack_, *this);
        this->release();
      } else {
        throw std::runtime_error(
            "the register is not in use, can't be stashed.");
      }
      return;
    }

    // save the content to stack
    void store() {
      if (in_use_) {
        ker_->mov(v_stack_, *this);
      } else {
        throw std::runtime_error(
            "the register is not in use, can't be stored.");
      }
      return;
    }

    // restore the content from stack, then aquire a register
    Xbyak::Reg64 restore() {
      if (!in_use_) {
        this->aquire();
        ker_->mov(*this, v_stack_);
      } else {
        throw std::runtime_error(
            "the register is already in use, can't be restored.");
      }
      return *this;
    }

    bool in_use_ = false;
    stack_var v_stack_;
    JitCodeGenerator* ker_;
  };

  struct vreg_var : public Xbyak::Ymm {
    explicit vreg_var(JitCodeGenerator* ker) : ker_(ker) {}

    ~vreg_var() {
      if (in_use_) this->release();
    }

    Xbyak::Ymm aquire() {
      if (!in_use_) {
        Xbyak::Ymm rf = this->ker_->get_free_vreg<Xbyak::Ymm>();
        setIdx(rf.getIdx());
        setKind(rf.getKind());
        setBit(rf.getBit());
        setOpmaskIdx(rf.getOpmaskIdx());
        setRounding(rf.getRounding());
        zero_ = rf.hasZero();
        in_use_ = true;
      } else {
        throw std::runtime_error("the vregister is already in use!");
      }
      return *this;
    }

    Xbyak::Ymm release() {
      if (in_use_) {
        in_use_ = false;
        this->ker_->drop_vreg(*this);
      } else {
        throw std::runtime_error(
            "the vregister is not in use, can't be relased.");
      }
      return *this;
    }

    Xbyak::Xmm xmm() { return Xbyak::Xmm(getIdx()); }

    bool in_use_ = false;
    JitCodeGenerator* ker_;
  };

  static void naive_impl(const float* a, const float* b, const size_t n,
                         float* c);

  using func_ptr_t = decltype(&JitCodeGenerator::naive_impl);

  virtual void dump_to_file(const char* fname = nullptr) {
    JitCodeGenerator* ptr = this;
    std::string default_fname_name =
        ptr->get_kernel_name() + std::string(".bin");
    if (fname == nullptr) {
      fname = default_fname_name.c_str();
    }
    // FILE* fp = tnn_fopen(fname, "w+");
    FILE* fp = nullptr;
    if (!fp) return;
    fwrite(get_func_ptr(), get_func_size(), 1, fp);
    fclose(fp);
  }

  virtual void* get_func_ptr() { return Xbyak::CodeArray::getCode<void*>(); }

  virtual size_t get_func_size() { return Xbyak::CodeArray::getSize(); }

  virtual std::string get_kernel_name() {
    return JIT_KERNEL_NAME(JitCodeGenerator);
  }

 protected:
  template <typename T>
  void declare_param() {
    if (sizeof(T) != abi::register_width_in_bytes) {
      throw std::runtime_error("unsupported param type.");
    }
    abi_nb_argment += 1;
  }

  void abi_prolog() {
    /* 1. save base stack pointer */
    push(xbp);
    mov(xbp, xsp);

    /*
        push(a) does the following things:
            rsp -= sizeof(void *)
            rsp = (a)
        so, after mov(xbp, xsp), [xbp] is the xbp of caller frame
        next pushed object will be at -4[xbp] or -8[xbp]
    */
    abi_bp_offset_ = -abi::register_width_in_bytes;
    /* 2. save the regs that abi require callee to save */
    /* only windows x64 need to save xmm6 - xmm15 regs */
#ifdef XBYAK64
    if (xmm_to_preserve) {
      sub(rsp, xmm_to_preserve * xmm_len);
      for (size_t i = 0; i < xmm_to_preserve; ++i) {
        movdqu(ptr[rsp + i * xmm_len], Xbyak::Xmm(xmm_to_preserve_start + i));
      }
    }
    abi_bp_offset_ -= xmm_to_preserve * xmm_len;
#endif

    for (int i = 0; i < abi::abi_nb_regs_callee_save; i++) {
      push(Xbyak::Reg64(abi::abi_regs_callee_save[i]));
      abi_bp_offset_ -= abi::register_width_in_bytes;
    }

    /* 3. save the arguements from register to stack */
    size_t abi_stack_arg_offset =
        abi::abi_stack_param_offset + 2 * abi::register_width_in_bytes;
    for (int i = 0; i < abi_nb_argment; i++) {
      if (i < abi::abi_nb_args_in_register) {
        push(Xbyak::Reg64(abi::abi_args_in_register[i]));
        arguement_offsets_.push_back(abi_bp_offset_);
        abi_bp_offset_ -= abi::register_width_in_bytes;
      } else {
        arguement_offsets_.push_back(abi_stack_arg_offset);
        abi_stack_arg_offset += abi::register_width_in_bytes;
      }
    }
  }

  void abi_epilog() {
    /* 1. rewind the stack for function arguments */
    size_t size_in_bytes = 0;
    for (int i = 0; i < abi_nb_argment; i++) {
      if (i < abi::abi_nb_args_in_register) {
        size_in_bytes += abi::register_width_in_bytes;
      }
    }
    if (size_in_bytes > 0) add(xsp, size_in_bytes);

    /* 2. restore the regs that abi require callee to save */
    for (int i = abi::abi_nb_regs_callee_save - 1; i >= 0; i--) {
      pop(Xbyak::Reg64(abi::abi_regs_callee_save[i]));
    }
#ifdef XBYAK64
    if (xmm_to_preserve) {
      for (size_t i = 0; i < xmm_to_preserve; ++i) {
        movdqu(Xbyak::Xmm(xmm_to_preserve_start + i), ptr[rsp + i * xmm_len]);
      }
      add(rsp, xmm_to_preserve * xmm_len);
    }
#endif
    /* 3. restore the base stack pointer */
    leave();  // mov(sp, bp); pop(bp);
  }

  Xbyak::Reg64 get_free_reg() {
    if (free_regs_.empty()) {
      throw std::runtime_error("not enough regs");
    }
    int id = free_regs_.front();
    // printf("total regs:%d get reg%d\n", free_regs_.size(), id);
    free_regs_.pop();
    return Xbyak::Reg64(common::regs_[id]);
  }

  void drop_reg(Xbyak::Reg64 r) {
    auto it = std::find(common::regs_.begin(), common::regs_.end(), r);
    int id = it - common::regs_.begin();
    // printf("drop reg%d\n", id);
    free_regs_.push(id);
  }

  template <typename T>
  T get_free_vreg() {
    if (free_vregs_.empty()) {
      throw std::runtime_error("not enough vector regs");
    }
    int id = free_vregs_.front();
    // printf("[xmm]total regs:%d get reg%d\n", free_vregs_.size(), id);
    free_vregs_.pop();
    return T(common::mmx_[id].getIdx());
  }

  template <typename T>
  void drop_vreg(T r) {
    Xbyak::Mmx mmx_r(r.getIdx());
    auto it = std::find(common::mmx_.begin(), common::mmx_.end(), mmx_r);
    int id = it - common::mmx_.begin();
    // printf("[xmm]drop reg%d\n", id);
    free_vregs_.push(id);
  }

  stack_var get_stack_var(bool clear = false,
                          size_t size_in_bytes = rword_.bit_ / 8) {
    std::vector<stack_var>::iterator candidate = stack_var_list_.end();
    for (auto it = stack_var_list_.begin(); it != stack_var_list_.end(); it++) {
      if (it->getBit() == size_in_bytes * 8) {
        candidate = it;
      }
    }

    stack_var ret(xsp, 0);
    if (candidate != stack_var_list_.end()) {
      ret = *candidate;
      stack_var_list_.erase(candidate);
    } else {
      stack_variable_offset_ += size_in_bytes;
      ret = stack_var(xsp, -stack_variable_offset_, size_in_bytes * 8);
    }
    if (clear) {
      Xbyak::Reg64 f = get_free_reg();
      xor_(f, f);
      mov(ret, f);
      drop_reg(f);
    }
    return ret;
  }

  void drop_stack_var(stack_var v) { stack_var_list_.push_back(v); }

  reg_var get_arguement(int i,
                        WHEN_TO_USE_FLAG when = WHEN_TO_USE_FLAG::LATER) {
    if (i >= arguement_offsets_.size()) {
      throw std::runtime_error("invalide arguement id.");
    }
    reg_var rf(this, stack_var(xbp, arguement_offsets_[i]));
    if (when == WHEN_TO_USE_FLAG::NOW) {
      rf.restore();
    }
    return rf;
  }

  stack_var get_arguement_to_stack(int i) {
    if (i >= arguement_offsets_.size()) {
      throw std::runtime_error("invalide arguement id.");
    }
    return stack_var(xbp, arguement_offsets_[i]);
  }

 public:
  JitCodeGenerator() : rword(rword_.bit_, rword_.broadcast_) {
    for (int i = 0; i < common::regs_.size(); i++) free_regs_.push(i);
    for (int i = 0; i < common::mmx_.size(); i++) free_vregs_.push(i);
  }

  virtual ~JitCodeGenerator() {}

  inline void vfmadd231ps_sse(Xbyak::Xmm v1, Xbyak::Xmm v2, Xbyak::Xmm v3) {
    if (cpu_with_isa(avx2)) {
      vfmadd231ps(v1, v2, v3);
    } else if (cpu_with_isa(sse42)) {
      // v2 * v3 -> v3
      // v1 + v3 -> v1
      mulps(v3, v2);
      addps(v1, v3);
    }
  }

  inline void vbroadcastss_sse(Xbyak::Xmm v, Xbyak::Address addr) {
    if (cpu_with_isa(avx2)) {
      vbroadcastss(v, addr);
    } else if (cpu_with_isa(sse42)) {
      movss(v, addr);
      shufps(v, v, 0);
    }
  }

  inline void vmovups_sse(Xbyak::Xmm v, Xbyak::Address addr) {
    if (cpu_with_isa(avx2)) {
      vmovups(v, addr);
    } else if (cpu_with_isa(sse42)) {
      movups(v, addr);
    }
  }

 protected:
  size_t abi_nb_argment = 0;
  size_t abi_bp_offset_ = 0;
  size_t stack_variable_offset_ = 0;

  std::vector<size_t> arguement_offsets_;
  std::vector<stack_var> stack_var_list_;

  Xbyak::Reg64 xbp = common::bp;
  Xbyak::Reg64 xsp = common::sp;

  std::queue<int> free_regs_;
  std::queue<int> free_vregs_;

  const Xbyak::AddressFrame rword;

  /* xmm regs need to be preserve */
  const size_t xmm_len = 16;
#ifdef _WIN32
  const size_t xmm_to_preserve_start = 6;
  const size_t xmm_to_preserve = 10;
#else
  const size_t xmm_to_preserve_start = 0;
  const size_t xmm_to_preserve = 0;
#endif
};

}  // namespace jit
}  // namespace cpu
}  // namespace kernels
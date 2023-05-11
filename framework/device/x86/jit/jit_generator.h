#pragma once

#include <xbyak/xbyak.h>

#include <algorithm>
#include <cmath>

#include "cpu_isa.h"

namespace kernels {
namespace cpu {
namespace jit {

constexpr int kMaxCodeSize = 256 * 1024;

class JITGenerator : public Xbyak::CodeGenerator {
 public:
  // Xbyak::CodeGenerator(size_t maxSize = DEFAULT_MAX_CODE_SIZE,
  //                      void *userPtr = 0, Allocator *allocator = 0)
  JITGenerator(size_t code_size = kMaxCodeSize)
      : Xbyak::CodeGenerator(code_size, Xbyak::AutoGrow) {}
  virtual ~JITGenerator() {}

  // TODO: add return status
  virtual void CreateKernel() {
    GenerateCode();
    jit_kernel_ = GetCode();
  }

  template <typename... KernelArgs>
  void operator()(KernelArgs... args) const {
    using jit_kernel_func_t = void (*)(const KernelArgs... args);
    auto *fptr = (jit_kernel_func_t)jit_kernel_;
    (*fptr)(std::forward<KernelArgs>(args)...);
  }

  template <typename... KernelArgs>
  void Forward(KernelArgs... args) const {
    using jit_kernel_func_t = void (*)(const KernelArgs... args);
    auto *fptr = (jit_kernel_func_t)jit_kernel_;
    (*fptr)(std::forward<KernelArgs>(args)...);
  }

  // We only consider system V ABI.
  void Preamble() {
    push(rbx);
    push(rbp);
    push(r12);
    push(r13);
    push(r14);
    push(r15);
  }

  void Postamble() {
    pop(r15);
    pop(r14);
    pop(r13);
    pop(r12);
    pop(rbp);
    pop(rbx);
    uni_vzeroupper();
    ret();
  }

  void uni_vzeroupper() {
    // if (cpu_has_isa(AVX)) vzeroupper();
  }

 protected:
  virtual void GenerateCode() = 0;
  const Xbyak::uint8 *jit_kernel_;

 private:
  const Xbyak::uint8 *GetCode() {
    // Xbyak::AutoGrow should call ready() before to run!
    // mode = Read/Write/Exec
    this->ready();
    const Xbyak::uint8 *code = CodeGenerator::getCode();
    return code;
  }
};

}  // namespace jit
}  // namespace cpu
}  // namespace kernels
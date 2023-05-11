#include "cpu_isa.h"

#include <stdio.h>
#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>

namespace kernels {
namespace cpu {
namespace jit {

using namespace Xbyak::util;

static Xbyak::util::Cpu cpu;

bool cpu_with_isa(x86_isa_t arch) {
  switch (arch) {
    case sse42:
      return cpu.has(Cpu::tSSE42);
    case avx:
      return cpu.has(Cpu::tAVX);
    case avx2:
      return cpu.has(Cpu::tAVX2) && cpu.has(Cpu::tFMA);
    case avx512:
      return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) &&
             cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ);
    case avx512_vnni:
      return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) &&
             cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ) &&
             cpu.has(Cpu::tAVX512_VNNI);
    default:
      return false;
  }
  return false;
}

}  // namespace jit
}  // namespace cpu
}  // namespace kernels

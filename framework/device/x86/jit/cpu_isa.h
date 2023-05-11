#pragma once

#include <immintrin.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>

#include <chrono>
#include <exception>
#include <fstream>
#include <random>

namespace kernels {
namespace cpu {
namespace jit {

typedef enum {
  sse42,
  avx,
  avx2,
  avx512,
  avx512_vnni,
} x86_isa_t;

bool cpu_with_isa(x86_isa_t arch);

}  // namespace jit
}  // namespace cpu
}  // namespace kernels

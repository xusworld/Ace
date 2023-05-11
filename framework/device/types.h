#pragma once

namespace ace {
namespace device {

enum class MemcpyKind {
  H2D,
  D2H,
  D2D,
  H2H,
};

enum class RuntimeType { CPU, CUDA };

}  // namespace device
}  // namespace ace
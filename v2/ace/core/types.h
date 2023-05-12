#pragma once

#include <string>

namespace ace {

enum class MemcpyKind {
  H2D,
  D2H,
  D2D,
  H2H,
};

enum class RuntimeType { CPU, CUDA };

inline std::string MemcpyKindToString(const MemcpyKind kind);
inline std::string RuntimeTypeToString(const RuntimeType type);
}  // namespace ace
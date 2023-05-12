#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

// type alias
using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;

using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

// Helper to delete copy constructor & copy-assignment operator
#define DISABLE_COPY_MOVE_ASSIGN(name)   \
  name(const name&) = delete;            \
  name& operator=(const name&) = delete; \
  name(name&&) = delete;                 \
  name& operator=(name&&) = delete

// Helper to declare copy constructor & copy-assignment operator default
#define DEFAULT_COPY_MOVE_ASSIGN(name)    \
  name(const name&) = default;            \
  name& operator=(const name&) = default; \
  name(name&&) = default;                 \
  name& operator=(name&&) = default


#ifndef macro_h
#define macro_h
#include <MNN/MNNDefine.h>

#define ALIMIN(x, y) ((x) < (y) ? (x) : (y))
#define ALIMAX(x, y) ((x) > (y) ? (x) : (y))

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)
#define ALIGN_UP8(x) ROUND_UP((x), 8)

// fraction length difference is 16bit. calculate the real value, it's about
// 0.00781
#define F32_BF16_MAX_LOSS ((0xffff * 1.0f) / (1 << 23))

// Helper to delete copy constructor & copy-assignment operator
#define DISABLE_COPY_MOVE_ASSIGN(name)   \
  name(const name&) = delete;            \
  name(name&&) = delete;                 \
  name& operator=(const name&) = delete; \
  name& operator=(name&&) = delete

// Helper to declare copy constructor & copy-assignment operator default
#define DEFAULT_COPY_MOVE_ASSIGN(name)    \
  name(const name&) = default;            \
  name(name&&) = default;                 \
  name& operator=(const name&) = default; \
  name& operator=(name&&) = default

#endif /* macro_h */

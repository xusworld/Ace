//
//  BF16Binary.cpp
//  MNN
//
//  Created by MNN on 2021/02/07.
//  Copyright © 2021, Alibaba Group Holding Limited
//

#include <algorithm>

#include "BF16Backend.hpp"
#include "BF16Binary.hpp"
#include "VecHalf.hpp"
#include "core/Macro.h"
#include "core/operation.h"
#include "device/cpu/BinaryUtils.hpp"
#include "math/Vec.hpp"
using Vec4Half = tars::Math::VecHalf<4>;
using Vec4 = tars::Math::Vec<float, 4>;
namespace tars {

template <typename Func>
void BF16BinaryWrap(void* dstRaw, const void* src0Raw, const void* src1Raw,
                    const int elementSize, const int needBroadcastIndex) {
  auto dst = (int16_t*)dstRaw;
  auto src0 = (int16_t*)src0Raw;
  auto src1 = (int16_t*)src1Raw;
  Func compute;
  const int sizeDivUnit = elementSize / 4;
  const int remainCount = elementSize - sizeDivUnit * 4;
  // FUNC_PRINT(needBroadcastIndex);

  float A[4];
  float B[4];
  float C[4];
  if (-1 == needBroadcastIndex) {
    if (sizeDivUnit > 0) {
      for (int i = 0; i < sizeDivUnit; ++i) {
        const auto src0Ptr = src0;
        const auto src1Ptr = src1;
        auto dstPtr = dst;
        Vec4::save(A, Vec4(std::move(Vec4Half::load(src0Ptr).value)));
        Vec4::save(B, Vec4(std::move(Vec4Half::load(src1Ptr).value)));
        for (int v = 0; v < 4; ++v) {
          C[v] = compute(A[v], B[v]);
        }
        Vec4Half::save(dstPtr, Vec4Half(std::move(Vec4::load(C).value)));
        src0 += 4;
        src1 += 4;
        dst += 4;
      }
    }
    if (remainCount > 0) {
      int16_t tempSrc0[4];
      int16_t tempSrc1[4];
      int16_t tempDst[4];
      ::memcpy(tempSrc0, src0, remainCount * sizeof(int16_t));
      ::memcpy(tempSrc1, src1, remainCount * sizeof(int16_t));
      Vec4::save(A, Vec4(std::move(Vec4Half::load(tempSrc0).value)));
      Vec4::save(B, Vec4(std::move(Vec4Half::load(tempSrc1).value)));
      for (int v = 0; v < remainCount; ++v) {
        C[v] = compute(A[v], B[v]);
      }
      Vec4Half::save(tempDst, Vec4Half(std::move(Vec4::load(C).value)));
      ::memcpy(dst, tempDst, remainCount * sizeof(int16_t));
    }
  } else if (0 == needBroadcastIndex) {
    const int16_t srcValue016 = src0[0];
    float srcValue0;
    BF16Functions::get()->MNNLowpToFp32(&srcValue016, &srcValue0, 1);
    if (sizeDivUnit > 0) {
      for (int i = 0; i < sizeDivUnit; ++i) {
        const auto src1Ptr = src1;
        auto dstPtr = dst;
        Vec4::save(B, Vec4(std::move(Vec4Half::load(src1Ptr).value)));
        for (int v = 0; v < 4; ++v) {
          C[v] = compute(srcValue0, B[v]);
        }
        Vec4Half::save(dstPtr, Vec4Half(std::move(Vec4::load(C).value)));
        src1 += 4;
        dst += 4;
      }
    }
    if (remainCount > 0) {
      int16_t tempSrc1[4];
      int16_t tempDst[4];
      ::memcpy(tempSrc1, src1, remainCount * sizeof(int16_t));
      Vec4::save(B, Vec4(std::move(Vec4Half::load(tempSrc1).value)));
      for (int v = 0; v < remainCount; ++v) {
        C[v] = compute(srcValue0, B[v]);
      }
      Vec4Half::save(tempDst, Vec4Half(std::move(Vec4::load(C).value)));
      ::memcpy(dst, tempDst, remainCount * sizeof(int16_t));
    }
  } else {
    const int16_t srcValue116 = src1[0];
    float srcValue1;
    BF16Functions::get()->MNNLowpToFp32(&srcValue116, &srcValue1, 1);
    if (sizeDivUnit > 0) {
      for (int i = 0; i < sizeDivUnit; ++i) {
        const auto src0Ptr = src0;
        auto dstPtr = dst;
        Vec4::save(A, Vec4(std::move(Vec4Half::load(src0Ptr).value)));
        for (int v = 0; v < 4; ++v) {
          C[v] = compute(A[v], srcValue1);
        }
        Vec4Half::save(dstPtr, Vec4Half(std::move(Vec4::load(C).value)));
        src0 += 4;
        dst += 4;
      }
    }
    if (remainCount > 0) {
      int16_t tempSrc0[4];
      int16_t tempDst[4];
      ::memcpy(tempSrc0, src0, remainCount * sizeof(int16_t));
      Vec4::save(A, Vec4(std::move(Vec4Half::load(tempSrc0).value)));
      for (int v = 0; v < remainCount; ++v) {
        C[v] = compute(A[v], srcValue1);
      }
      Vec4Half::save(tempDst, Vec4Half(std::move(Vec4::load(C).value)));
      ::memcpy(dst, tempDst, remainCount * sizeof(int16_t));
    }
  }
}

template <typename Func>
void BF16Binary(void* dstRaw, const void* src0Raw, const void* src1Raw,
                const int elementSize, const int needBroadcastIndex) {
  auto dst = (int16_t*)dstRaw;
  auto src0 = (int16_t*)src0Raw;
  auto src1 = (int16_t*)src1Raw;
  Func compute;
  const int sizeDivUnit = elementSize / 4;
  const int remainCount = elementSize - sizeDivUnit * 4;

  if (-1 == needBroadcastIndex) {
    if (sizeDivUnit > 0) {
      for (int i = 0; i < sizeDivUnit; ++i) {
        Vec4Half a = Vec4Half::load(src0);
        Vec4Half b = Vec4Half::load(src1);
        Vec4Half::save(dst, compute(a, b));
        src0 += 4;
        src1 += 4;
        dst += 4;
      }
    }
    if (remainCount > 0) {
      int16_t tempSrc0[4];
      int16_t tempSrc1[4];
      int16_t tempDst[4];
      ::memcpy(tempSrc0, src0, remainCount * sizeof(int16_t));
      ::memcpy(tempSrc1, src1, remainCount * sizeof(int16_t));
      Vec4Half a = Vec4Half::load(tempSrc0);
      Vec4Half b = Vec4Half::load(tempSrc1);
      Vec4Half::save(tempDst, compute(a, b));
      ::memcpy(dst, tempDst, remainCount * sizeof(int16_t));
    }
  } else if (0 == needBroadcastIndex) {
    const int16_t srcValue016 = src0[0];
    float srcValue0;
    BF16Functions::get()->MNNLowpToFp32(&srcValue016, &srcValue0, 1);
    Vec4Half a = Vec4Half(srcValue0);
    if (sizeDivUnit > 0) {
      for (int i = 0; i < sizeDivUnit; ++i) {
        const auto src1Ptr = src1;
        auto dstPtr = dst;
        Vec4Half b = Vec4Half::load(src1Ptr);
        Vec4Half::save(dstPtr, compute(a, b));
        src1 += 4;
        dst += 4;
      }
    }
    if (remainCount > 0) {
      int16_t tempSrc1[8];
      int16_t tempDst[8];
      ::memcpy(tempSrc1, src1, remainCount * sizeof(int16_t));
      Vec4Half b = Vec4Half::load(tempSrc1);
      Vec4Half::save(tempDst, compute(a, b));
      ::memcpy(dst, tempDst, remainCount * sizeof(int16_t));
    }
  } else {
    const int16_t srcValue116 = src1[0];
    float srcValue1;
    BF16Functions::get()->MNNLowpToFp32(&srcValue116, &srcValue1, 1);
    Vec4Half b = Vec4Half(srcValue1);
    if (sizeDivUnit > 0) {
      for (int i = 0; i < sizeDivUnit; ++i) {
        const auto src0Ptr = src0;
        auto dstPtr = dst;
        Vec4Half a = Vec4Half::load(src0Ptr);
        Vec4Half::save(dstPtr, compute(a, b));
        src0 += 4;
        dst += 4;
      }
    }
    if (remainCount > 0) {
      int16_t tempSrc0[8];
      int16_t tempDst[8];
      ::memcpy(tempSrc0, src0, remainCount * sizeof(int16_t));
      Vec4Half a = Vec4Half::load(tempSrc0);
      Vec4Half::save(tempDst, compute(a, b));
      ::memcpy(dst, tempDst, remainCount * sizeof(int16_t));
    }
  }
}

struct VecBinaryAdd : std::binary_function<Vec4Half, Vec4Half, Vec4Half> {
  Vec4Half operator()(const Vec4Half& x, const Vec4Half& y) const {
    return x + y;
  }
};

struct VecBinarySub : std::binary_function<Vec4Half, Vec4Half, Vec4Half> {
  Vec4Half operator()(const Vec4Half& x, const Vec4Half& y) const {
    return x - y;
  }
};

struct VecBinaryMul : std::binary_function<Vec4Half, Vec4Half, Vec4Half> {
  Vec4Half operator()(const Vec4Half& x, const Vec4Half& y) const {
    return x * y;
  }
};

struct VecBinaryMin : std::binary_function<Vec4Half, Vec4Half, Vec4Half> {
  Vec4Half operator()(const Vec4Half& x, const Vec4Half& y) const {
    return Vec4Half::min(x, y);
  }
};

struct VecBinaryMax : std::binary_function<Vec4Half, Vec4Half, Vec4Half> {
  Vec4Half operator()(const Vec4Half& x, const Vec4Half& y) const {
    return Vec4Half::max(x, y);
  }
};

struct VecBinarySqd : std::binary_function<Vec4Half, Vec4Half, Vec4Half> {
  Vec4Half operator()(const Vec4Half& x, const Vec4Half& y) const {
    return (x - y) * (x - y);
  }
};

MNNBinaryExecute BF16BinaryFloatSelect(int type) {
  switch (type) {
    case BinaryOpOperation_ADD:
      return BF16Binary<VecBinaryAdd>;
      break;
    case BinaryOpOperation_SUB:
      return BF16Binary<VecBinarySub>;
      break;
    case BinaryOpOperation_MUL:
      return BF16Binary<VecBinaryMul>;
      break;
    case BinaryOpOperation_MINIMUM:
      return BF16Binary<VecBinaryMin>;
      break;
    case BinaryOpOperation_MAXIMUM:
      return BF16Binary<VecBinaryMax>;
      break;
    case BinaryOpOperation_SquaredDifference:
      return BF16Binary<VecBinarySqd>;
      break;
    case BinaryOpOperation_REALDIV:
      return BF16BinaryWrap<BinaryRealDiv<float, float, float>>;
      break;
    case BinaryOpOperation_FLOORDIV:
      return BF16BinaryWrap<BinaryFloorDiv<float, float, float>>;
      break;
    case BinaryOpOperation_FLOORMOD:
      return BF16BinaryWrap<BinaryFloorMod<float, float, float>>;
      break;
    case BinaryOpOperation_POW:
      return BF16BinaryWrap<BinaryPow<float, float, float>>;
      break;
    case BinaryOpOperation_ATAN2:
      return BF16BinaryWrap<BinaryAtan2<float, float, float>>;
      break;
    case BinaryOpOperation_MOD:
      return BF16BinaryWrap<BinaryMod<float, float, float>>;
      break;
    default:
      return nullptr;
      break;
  }
  return nullptr;
}

}  // namespace tars

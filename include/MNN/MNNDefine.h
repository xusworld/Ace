//
//  MNNDefine.h
//  MNN
//
//  Created by MNN on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDefine_h
#define MNNDefine_h

#include <assert.h>
#include <stdio.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define MNN_BUILD_FOR_IOS
#endif
#endif

#define MNN_PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define MNN_ERROR(format, ...) printf(format, ##__VA_ARGS__)

#ifdef DEBUG
#define MNN_ASSERT(x)                                      \
  {                                                        \
    int res = (x);                                         \
    if (!res) {                                            \
      MNN_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
      assert(res);                                         \
    }                                                      \
  }
#else
#define MNN_ASSERT(x)
#endif

#define FUNC_PRINT(x) MNN_PRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define FUNC_PRINT_ALL(x, type) \
  MNN_PRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#define MNN_CHECK(success, log)                             \
  if (!(success)) {                                         \
    MNN_ERROR("Check failed: %s ==> %s\n", #success, #log); \
  }

#if defined(_MSC_VER)
#if defined(BUILDING_MNN_DLL)
#define MNN_PUBLIC __declspec(dllexport)
#elif defined(USING_MNN_DLL)
#define MNN_PUBLIC __declspec(dllimport)
#else
#define MNN_PUBLIC
#endif
#else
#define MNN_PUBLIC __attribute__((visibility("default")))
#endif
#define STR_IMP(x) #x
#define STR(x) STR_IMP(x)
#define MNN_VERSION_MAJOR 2
#define MNN_VERSION_MINOR 4
#define MNN_VERSION_PATCH 0
#define MNN_VERSION \
  STR(MNN_VERSION_MAJOR) "." STR(MNN_VERSION_MINOR) "." STR(MNN_VERSION_PATCH)
#endif /* MNNDefine_h */

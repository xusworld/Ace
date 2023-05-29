//
//  Concurrency.h
//  MNN
//
//  Created by MNN on 2018/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef concurrency_h
#define concurrency_h

#ifdef MNN_FORBIT_MULTI_THREADS
#define MNN_CONCURRENCY_BEGIN(__iter__, __num__) \
  for (int __iter__ = 0; __iter__ < __num__; __iter__++) {
#define MNN_CONCURRENCY_END() }

#elif defined(MNN_USE_THREAD_POOL)
#include "device/cpu/ThreadPool.hpp"

#define MNN_STRINGIFY(a) #a
#define MNN_CONCURRENCY_BEGIN(__iter__, __num__)   \
  {                                                \
    std::pair<std::function<void(int)>, int> task; \
    task.second = __num__;                         \
    task.first = [&](int __iter__) {
#define MNN_CONCURRENCY_END()                                     \
  }                                                               \
  ;                                                               \
  auto cpuBn = (CPUDevice*)backend();                             \
  tars::ThreadPool::enqueue(std::move(task), cpuBn->taskIndex()); \
  }

#endif
#endif /* concurrency_h */
